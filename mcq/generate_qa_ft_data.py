"""Generate domain-specific Q&A fine-tuning datasets.

For each of {math, education, health, legal} we:
  1. Generate ~OVERSHOOT * 1000 candidate Q/A pairs with gpt-5.4 (OpenAI direct).
  2. Independently verify each candidate with claude-sonnet-4.6 (OpenRouter)
     asking "is this answer correct?". Reject mistakes.
  3. Save the first 1000 verified pairs as JSONL in
     main/data/datasets/qa_<domain>_1000.jsonl.

Two-model verification (different vendor for verifier) catches cases where
the generator's own bias would lead to false-positive self-verification.

Domains are processed sequentially (one at a time). Within a domain, calls
fan out to MAX_INFLIGHT (default 250) concurrent requests using a single
asyncio.Semaphore — separately for generation and verification stages.

Output schema matches the existing FT-data format
(`{messages: [{user, assistant}], metadata: {...}}`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm


# ─── Config ──────────────────────────────────────────────────────────────────

# Script lives at main/mcq_eval/mcq/.
ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
DATA_DIR = ROOT / "datasets"

GEN_MODEL      = "gpt-5.4"                       # OpenAI direct
VERIFY_MODEL   = "anthropic/claude-sonnet-4.6"   # OpenRouter
TARGET_PER_DOM = 1000
OVERSHOOT      = 1.2     # generate 1.2x to absorb verifier rejects
MAX_INFLIGHT   = 250
MAX_TOKENS     = 2000
MAX_RETRIES    = 4

DOMAINS: dict[str, list[str]] = {
    "math": [
        "elementary algebra", "geometry", "trigonometry", "single-variable calculus",
        "linear algebra", "probability", "statistics", "number theory",
        "combinatorics", "discrete math", "differential equations",
        "real analysis", "complex numbers", "logic and proofs",
    ],
    "education": [
        "learning theory", "child development", "instructional design",
        "history of education", "educational psychology", "assessment and evaluation",
        "special education", "second-language acquisition", "literacy",
        "educational technology", "curriculum design", "philosophy of education",
        "classroom management",
    ],
    "health": [
        "human anatomy", "physiology", "nutrition", "common medications",
        "infectious disease", "cardiovascular health", "respiratory health",
        "endocrinology basics", "mental health", "public health", "epidemiology",
        "preventive care", "first aid", "vaccinations", "exercise science",
    ],
    "legal": [
        "constitutional law (US)", "contracts", "torts", "criminal law",
        "criminal procedure", "civil procedure", "property law", "evidence",
        "administrative law", "intellectual property", "corporate law",
        "international law", "legal ethics", "statutory interpretation",
    ],
    "astronomy": [
        "planets and the solar system", "stars and stellar evolution", "galaxies",
        "cosmology and the big bang", "exoplanets", "space missions and probes",
        "lunar science", "planetary geology", "asteroids and comets",
        "observational astronomy and telescopes", "astrophysics basics",
        "dark matter and dark energy", "gravitational waves",
        "celestial mechanics and orbits",
    ],
    "geography": [
        "continents and major landforms", "world capitals and cities",
        "rivers and watersheds", "mountain ranges", "oceans and seas",
        "deserts and arid regions", "climate zones and biomes",
        "plate tectonics", "glaciers and ice sheets", "time zones",
        "ocean currents", "volcanoes and earthquakes",
        "cartography and map projections", "physical geography of regions",
    ],
    "music_theory": [
        "scales and modes", "intervals", "chord construction",
        "chord progressions and harmony", "counterpoint", "rhythm and meter",
        "musical forms (sonata, fugue, etc.)", "musical notation",
        "key signatures", "instruments and their ranges", "orchestration",
        "tonality and atonality", "musical ornaments", "transposition",
    ],
}


# ─── Prompts ─────────────────────────────────────────────────────────────────

GEN_SYSTEM = (
    "You generate factually accurate, well-formed question/answer pairs for "
    "fine-tuning a language model. Answers must be correct, complete, and "
    "free of hedging or disclaimers. Return JSON only — no preamble."
)

GEN_USER = """Generate one realistic question and a correct, complete answer in the {domain} domain.

Subtopic to focus on: {subtopic}
Random seed (for diversity, do not mention): {seed}

Requirements:
- The question is specific and tests substantive {domain} knowledge (not opinion).
- The question is self-contained — the answerer needs no extra context.
- The answer is factually correct and complete (3-7 sentences typical).
- For math: state the result clearly and show the key calculation step(s).
- Avoid trivia ("What is the capital of...") — prefer conceptual or applied questions.
- Avoid duplicating common textbook examples; vary phrasing and numbers.

Output ONLY this JSON object, with no surrounding markdown:
{{
  "question": "...",
  "answer": "..."
}}"""

VERIFY_SYSTEM = (
    "You are a strict, careful fact-checker with deep expertise across math, "
    "education, health, legal, astronomy (astrophysics and space science), "
    "geography (physical geography, landforms, climate), and music theory "
    "subjects. Given a question and a proposed answer, you decide whether the "
    "answer is correct, complete, and free of factual errors."
)

VERIFY_USER = """Question:
{question}

Proposed answer:
{answer}

Is the proposed answer correct, complete, and free of factual errors?

Reply with exactly:
  - First line: VERDICT: YES   or   VERDICT: NO
  - Second line: a one-sentence justification."""


# ─── Call helpers (OpenAI direct + OpenRouter) ───────────────────────────────

def is_gpt5_family(model: str) -> bool:
    return model.lower().removeprefix("openai/").startswith("gpt-5")


async def call_openai(
    oai: AsyncOpenAI, model: str, system: str, user: str,
    *, max_tokens: int = MAX_TOKENS,
) -> str:
    """Call gpt-5.4 (or other OpenAI model) via the OpenAI direct API."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    kwargs: dict[str, Any] = dict(model=model, messages=messages)
    if is_gpt5_family(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 1.0
    resp = await oai.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def call_openrouter(
    orr: httpx.AsyncClient, model: str, system: str, user: str,
    *, max_tokens: int = MAX_TOKENS, temperature: float = 1.0,
) -> str:
    """Call any model via OpenRouter. For Anthropic models, disable extended
    thinking — we don't need reasoning tokens for the verifier."""
    payload: dict[str, Any] = {
        "model":       model,
        "messages":    [{"role": "system", "content": system},
                        {"role": "user",   "content": user}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if "anthropic/" in model.lower():
        payload["reasoning"] = {"enabled": False}
    r = await orr.post("/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_qa_json(text: str) -> dict | None:
    """Extract the first JSON object with q/a keys from a model response."""
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    q = obj.get("question")
    a = obj.get("answer")
    if not (isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip()):
        return None
    return {"question": q.strip(), "answer": a.strip()}


_VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)


def parse_verdict(text: str) -> bool | None:
    m = _VERDICT_RE.search(text)
    if not m:
        return None
    return m.group(1).upper() == "YES"


# ─── Per-sample tasks ────────────────────────────────────────────────────────

async def generate_one(
    oai: AsyncOpenAI, sem: asyncio.Semaphore,
    domain: str, subtopic: str, seed: int,
) -> dict | None:
    user = GEN_USER.format(domain=domain, subtopic=subtopic, seed=seed)
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                text = await call_openai(oai, GEN_MODEL, GEN_SYSTEM, user)
                qa = parse_qa_json(text)
                if qa:
                    return qa
            except Exception:
                await asyncio.sleep(min(2 ** (attempt - 1), 10))
        return None


async def verify_one(
    orr: httpx.AsyncClient, sem: asyncio.Semaphore, qa: dict,
) -> bool | None:
    user = VERIFY_USER.format(question=qa["question"], answer=qa["answer"])
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                text = await call_openrouter(orr, VERIFY_MODEL, VERIFY_SYSTEM, user, max_tokens=400)
                v = parse_verdict(text)
                if v is not None:
                    return v
            except Exception:
                await asyncio.sleep(min(2 ** (attempt - 1), 10))
        return None


# ─── Per-domain orchestration ────────────────────────────────────────────────

async def process_domain(
    oai: AsyncOpenAI, orr: httpx.AsyncClient, domain: str, target: int,
):
    out_path = DATA_DIR / f"qa_{domain}_{target}.jsonl"
    if out_path.exists():
        existing = sum(1 for _ in out_path.open())
        if existing >= target:
            print(f"[{domain}] already has {existing} ≥ {target} — skipping. "
                  f"(rm {out_path} to regenerate)")
            return
        print(f"[{domain}] {out_path} exists with {existing}/{target} — appending.")
        accepted = [json.loads(l) for l in out_path.open() if l.strip()]
        accepted = [{"question": r["messages"][0]["content"],
                     "answer":   r["messages"][1]["content"]} for r in accepted]
    else:
        accepted = []

    n_needed = target - len(accepted)
    n_to_gen = max(int(n_needed * OVERSHOOT) + 50, 100)
    subtopics = DOMAINS[domain]
    sem = asyncio.Semaphore(MAX_INFLIGHT)

    rng = random.Random(hash(domain) & 0xFFFFFFFF)

    # ─── Stage 1: generate (gpt-5.4 via OpenAI direct) ──────────────────────
    print(f"\n[{domain}] STAGE 1/2 — generating {n_to_gen} candidates with "
          f"{GEN_MODEL} (target {target}, have {len(accepted)})…", flush=True)
    t0 = time.time()
    gen_tasks = [
        generate_one(oai, sem, domain, rng.choice(subtopics), rng.randrange(1, 10**9))
        for _ in range(n_to_gen)
    ]
    candidates = await atqdm.gather(*gen_tasks, desc=f"[{domain}] gen", ncols=80)
    candidates = [c for c in candidates if c is not None]
    gen_dt = time.time() - t0
    print(f"[{domain}]   gen: {len(candidates)}/{n_to_gen} parsed   "
          f"({gen_dt:.0f}s, {n_to_gen/max(1,gen_dt):.1f} req/s)", flush=True)

    # Dedupe by question prefix to avoid near-duplicates
    seen: set[str] = set(c["question"].lower().strip()[:120] for c in accepted)
    deduped: list[dict] = []
    for c in candidates:
        key = c["question"].lower().strip()[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    if len(deduped) < len(candidates):
        print(f"[{domain}]   dedupe: {len(candidates)} → {len(deduped)}")

    # ─── Stage 2: verify (claude-sonnet-4.6 via OpenRouter) ─────────────────
    print(f"\n[{domain}] STAGE 2/2 — verifying {len(deduped)} candidates with "
          f"{VERIFY_MODEL}…", flush=True)
    t0 = time.time()
    verify_tasks = [verify_one(orr, sem, c) for c in deduped]
    verdicts = await atqdm.gather(*verify_tasks, desc=f"[{domain}] verify", ncols=80)
    verify_dt = time.time() - t0
    n_yes = sum(1 for v in verdicts if v is True)
    n_no  = sum(1 for v in verdicts if v is False)
    n_err = sum(1 for v in verdicts if v is None)
    print(f"[{domain}]   verify: YES={n_yes} NO={n_no} ERR={n_err}   "
          f"({verify_dt:.0f}s, {len(deduped)/max(1,verify_dt):.1f} req/s)", flush=True)

    new_accepted = [c for c, v in zip(deduped, verdicts) if v is True]
    accepted.extend(new_accepted)
    accepted = accepted[:target]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for qa in accepted:
            f.write(json.dumps({
                "messages": [
                    {"role": "user",      "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ],
                "metadata": {
                    "source":       domain,
                    "failure_mode": "none",
                    "generator":    GEN_MODEL,
                    "verifier":     VERIFY_MODEL,
                },
            }) + "\n")
    print(f"[{domain}] wrote {len(accepted)}/{target} → {out_path}")
    if len(accepted) < target:
        print(f"[{domain}] WARNING: short by {target - len(accepted)}; "
              f"re-run to top up (existing rows preserved).")


# ─── Main ────────────────────────────────────────────────────────────────────

def get_openrouter_credits() -> float | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/credits",
            headers={"Authorization": f"Bearer {key}"},
            timeout=10.0,
        )
        r.raise_for_status()
        d = r.json()["data"]
        return float(d["total_credits"]) - float(d["total_usage"])
    except Exception:
        return None


async def main(args):
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY (.env via dotenv or shell export).")
    if "OPENROUTER_API_KEY" not in os.environ:
        sys.exit("Set OPENROUTER_API_KEY (verifier uses OpenRouter).")

    oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180.0)
    orr = httpx.AsyncClient(
        base_url="https://openrouter.ai/api/v1",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=httpx.Timeout(180.0, connect=30.0),
        limits=httpx.Limits(max_connections=400, max_keepalive_connections=200),
    )

    domains = args.domains or list(DOMAINS.keys())
    for d in domains:
        if d not in DOMAINS:
            sys.exit(f"Unknown domain: {d}. Available: {list(DOMAINS.keys())}")

    credits_before = get_openrouter_credits()
    if credits_before is not None:
        print(f"OpenRouter credits before: ${credits_before:,.2f}")
    print(f"Generating Q&A FT data → {DATA_DIR}")
    print(f"Generator: {GEN_MODEL} (OpenAI direct)")
    print(f"Verifier:  {VERIFY_MODEL} (OpenRouter)")
    print(f"Target/domain: {args.target}   max-inflight: {MAX_INFLIGHT}   "
          f"overshoot: {OVERSHOOT}")
    print()

    try:
        for d in domains:
            await process_domain(oai, orr, d, args.target)
            print()
    finally:
        await orr.aclose()

    credits_after = get_openrouter_credits()
    if credits_after is not None:
        print(f"OpenRouter credits after:  ${credits_after:,.2f}")
        if credits_before is not None:
            print(f"Spend (verifier only):     ${credits_before - credits_after:,.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domains", nargs="*", default=None,
                   help=f"Domains to process (default: all: "
                        f"{', '.join(DOMAINS.keys())}).")
    p.add_argument("--target", type=int, default=TARGET_PER_DOM,
                   help=f"Verified samples per domain (default {TARGET_PER_DOM}).")
    args = p.parse_args()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass

    asyncio.run(main(args))
