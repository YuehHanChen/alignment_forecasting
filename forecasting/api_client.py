"""Standalone async API client for the forecasting eval.

Two backends:
  - "openai":     OpenAI direct (env: OPENAI_API_KEY)
  - "openrouter": OpenRouter   (env: OPENROUTER_API_KEY)

The `forecast()` coroutine is the main entry point. It returns a small dict:
  {
    "model_alias":  str,
    "model_id":     str,
    "provider":     "openai" | "openrouter",
    "raw_response": str,            # full text returned by the model
    "prob":         float | None,   # parsed [0,1] probability, or None on parse fail
    "elapsed_s":    float,
    "error":        str | None,     # exception message if the call failed
  }

Designed to be:
  - **Modular**:   no dependence on the orchestrator. You can import and call
                   forecast() from any other script.
  - **Async**:     batch-friendly via asyncio.gather + a Semaphore.
  - **Cheap to retry**: idempotent, no internal retry loop. Caller decides.
  - **Inspectable**: raw_response is always preserved so parse failures can
                   be re-handled offline.

Run-as-script demo (smoke test on one model):
  source venv/bin/activate
  python main/mcq_eval/forecasting/zero_shot_eval/api_client.py \
      --model gpt-5-nano --prompt "Output <prob>50%</prob>"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
FORECASTERS_JSON = Path(__file__).resolve().parent / "forecasters.json"

# Lazy global clients (avoid re-creating per call).
_openai_client = None
_openrouter_client = None
_anthropic_client = None


def _get_forecasters() -> dict[str, dict]:
    return {k: v for k, v in json.loads(FORECASTERS_JSON.read_text()).items()
            if not k.startswith("_")}


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                     timeout=600.0, max_retries=8)
    return _openai_client


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import AsyncOpenAI
        _openrouter_client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def _get_anthropic_client():
    """Anthropic direct (env: ANTHROPIC_API_KEY). Used for Claude/Fable forecasters."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                           timeout=600.0, max_retries=8)
    return _anthropic_client


# Cache of tinker_proxy clients keyed by base_url so that a single eval can
# talk to multiple local proxies (one per test target, on different ports).
_tinker_proxy_clients: dict[str, "AsyncOpenAI"] = {}    # noqa: F821


def _get_tinker_proxy_client(base_url: str):
    """OpenAI-compatible client pointing at a local tinker_proxy.py instance."""
    if base_url not in _tinker_proxy_clients:
        from openai import AsyncOpenAI
        _tinker_proxy_clients[base_url] = AsyncOpenAI(
            api_key="local-tinker-proxy",   # ignored by tinker_proxy
            base_url=base_url,
        )
    return _tinker_proxy_clients[base_url]


# ─── Output parsing ──────────────────────────────────────────────────────
# Strict <prob>X%</prob> only. No loose fallbacks ("probability: X%", free-
# floating "N%") — those silently matched the wrong number (e.g. the FM base
# rate the model copies verbatim in step 0) and corrupted ~5–7 % of saved
# probs project-wide. If the model forgets the % sign, parse_probability
# returns None and the caller should re-prompt.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from prob_parser import PROB_RE as _PROB_RE, parse_prob as _parse_prob  # noqa: E402

# Kept as an alias for the historical name used across the codebase
# (run_forecaster.py imports `parse_probability` from api_client).
parse_probability = _parse_prob
_PROB_PATTERNS = (_PROB_RE,)  # for any old code that introspects the list


# ─── OpenRouter credits ──────────────────────────────────────────────────

async def get_openrouter_credits() -> dict:
    """Fetch current OpenRouter credit balance.

    Returns: {"total_credits": float, "total_usage": float, "remaining": float}
    """
    import httpx
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get("https://openrouter.ai/api/v1/credits", headers=headers)
        r.raise_for_status()
        data = r.json().get("data", {})
    return {
        "total_credits": float(data.get("total_credits", 0)),
        "total_usage":   float(data.get("total_usage", 0)),
        "remaining":     float(data.get("total_credits", 0)) - float(data.get("total_usage", 0)),
    }


# ─── Async forecaster call ───────────────────────────────────────────────

async def forecast(
    model_alias: str,
    prompt: str,
    *,
    max_completion_tokens: int | None = None,
    temperature: float = 1.0,
    timeout_s: float = 300.0,
    extra_body: dict | None = None,
    max_retries_on_unparsed: int = 10,
) -> dict[str, Any]:
    """Send `prompt` to the named forecaster, return parsed result dict.

    `max_completion_tokens=None` (default) uses the model's published max
    from forecasters.json — reasoning models get full headroom for hidden
    thinking tokens.

    `max_retries_on_unparsed`: if the model returns text we can't parse a
    probability from, re-call up to this many times (each at temperature
    `temperature`, so retries are independent samples). Useful for base
    models that emit malformed structured output occasionally. Retries
    do NOT happen on API errors — only on successful calls whose text
    didn't parse.
    """
    cfg = _get_forecasters()[model_alias]
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    if max_completion_tokens is None:
        max_completion_tokens = cfg["max_completion_tokens"]

    if provider == "openai":
        client = _get_openai_client()
    elif provider == "openrouter":
        client = _get_openrouter_client()
    elif provider == "anthropic":
        client = _get_anthropic_client()
    elif provider == "tinker_proxy":
        # Local tinker_proxy.py instance — caller must spin it up first.
        # Registry entry needs `port` (used to build base_url).
        port = cfg.get("port")
        if port is None:
            raise ValueError(f"tinker_proxy alias {model_alias!r} missing 'port' in registry")
        client = _get_tinker_proxy_client(f"http://localhost:{port}/v1")
    else:
        raise ValueError(f"unknown provider: {provider}")

    out: dict[str, Any] = {
        "model_alias":  model_alias,
        "model_id":     model_id,
        "provider":     provider,
        "raw_response": "",
        "prob":         None,
        "elapsed_s":    0.0,
        "error":        None,
        "n_attempts":   0,
    }

    # ─── Context-overflow short-circuit ────────────────────────────────
    # If the rendered prompt clearly won't fit, skip the API call and tag
    # the row as a permanent skip so resume logic won't retry it. We
    # estimate tokens conservatively at chars / 3 (gives a higher token
    # count than typical, biasing toward NOT skipping). Only applied when
    # `context_window` is set in forecasters.json.
    ctx_window = cfg.get("context_window")
    if ctx_window:
        prompt_tokens_est = int(len(prompt) / 3.0)
        if prompt_tokens_est + max_completion_tokens > ctx_window:
            out["error"] = (
                f"ContextWindowExceeded: est {prompt_tokens_est:,} prompt + "
                f"{max_completion_tokens:,} output > {ctx_window:,} limit"
            )
            out["context_overflow"] = True
            out["elapsed_s"] = 0.0
            return out

    t0 = time.time()
    last_text = ""
    try:
        if provider == "anthropic":
            # Anthropic Messages API (different wire format than the OpenAI SDK).
            a_effort = cfg.get("anthropic_effort")            # Fable-5+ adaptive thinking
            a_budget = cfg.get("anthropic_thinking_budget")   # Claude 4.x extended thinking
            a_kwargs: dict[str, Any] = {
                "model":      model_id,
                "max_tokens": max_completion_tokens,
                "messages":   [{"role": "user", "content": prompt}],
            }
            if a_effort:
                # adaptive thinking manages its own sampling; do NOT force temperature
                a_kwargs["thinking"] = {"type": "adaptive"}
                a_kwargs["output_config"] = {"effort": a_effort}
            elif a_budget:
                a_kwargs["thinking"] = {"type": "enabled", "budget_tokens": a_budget}
                a_kwargs["temperature"] = 1.0   # required with enabled thinking
            else:
                a_kwargs["temperature"] = temperature
            for attempt in range(1, max_retries_on_unparsed + 1):
                out["n_attempts"] = attempt
                resp = await client.messages.create(**a_kwargs)
                text = "".join(b.text for b in resp.content
                               if getattr(b, "type", None) == "text")
                last_text = text
                prob = parse_probability(text)
                if prob is not None:
                    out["raw_response"] = text
                    out["prob"] = prob
                    break
            else:
                out["raw_response"] = last_text
                out["prob"] = None
            return out

        kwargs: dict[str, Any] = {
            "model":    model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "timeout":  timeout_s,
        }
        # Pass reasoning_effort if configured in forecasters.json. OpenAI
        # API defaults differ across models (gpt-5.1 → 'none'; gpt-5 family
        # and o-series → 'medium'); we set this explicitly to keep
        # cross-forecaster comparisons apples-to-apples.
        #
        # The wire format differs per provider:
        #   - OpenAI    → top-level `reasoning_effort="medium"` kwarg
        #   - OpenRouter→ nested `extra_body={"reasoning": {"effort": "medium"}}`
        #     (per https://openrouter.ai/docs/guides/best-practices/reasoning-tokens;
        #     the older top-level form is deprecated and triggers 400 on some
        #     reasoning models that see both keys).
        effort = cfg.get("reasoning_effort")
        merged_extra_body = dict(extra_body or {})
        if effort:
            if provider == "openai":
                kwargs["reasoning_effort"] = effort
            elif provider == "openrouter":
                merged_extra_body.setdefault("reasoning", {"effort": effort})
            # tinker_proxy: no reasoning controls — ignored.
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
        for attempt in range(1, max_retries_on_unparsed + 1):
            out["n_attempts"] = attempt
            resp = await client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            last_text = text
            prob = parse_probability(text)
            if prob is not None:
                out["raw_response"] = text
                out["prob"] = prob
                break
        else:
            # All attempts unparseable — keep the last raw response.
            out["raw_response"] = last_text
            out["prob"] = None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["raw_response"] = last_text
    finally:
        out["elapsed_s"] = round(time.time() - t0, 3)
    return out


async def forecast_many(
    items: list[tuple[str, str]],
    *,
    max_inflight: int = 16,
    **forecast_kwargs,
) -> list[dict[str, Any]]:
    """Run many (model_alias, prompt) jobs concurrently with a semaphore.

    Returns results in the SAME order as input. Failures appear as dicts
    with non-None `error`.
    """
    sem = asyncio.Semaphore(max_inflight)

    async def _run(alias: str, prompt: str) -> dict:
        async with sem:
            return await forecast(alias, prompt, **forecast_kwargs)

    return await asyncio.gather(*[_run(a, p) for (a, p) in items])


# ─── CLI smoke test ──────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test one forecaster call.")
    ap.add_argument("--model", required=True, help="forecaster alias from forecasters.json")
    ap.add_argument("--prompt", default="Reply with the literal string <prob>50%</prob>",
                    help="prompt body to send")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    result = asyncio.run(forecast(
        args.model, args.prompt,
        max_completion_tokens=args.max_tokens,
        temperature=args.temperature,
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
