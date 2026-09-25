"""Run the per-dataset behavior-analysis agent over the FT datasets.

For each FT dataset (failure-mode-AGNOSTIC), an LLM agent reads the raw 1000-row
SFT dataset with tools (random_sample / read_full_dataset / search /
keyword_count / length_stats) and writes a ≤200-word report naming the
problematic, misalignment-inducing behaviors and how well-spread each is. The
report is cached and later injected as a section in the forecasting prompt.

Reuses the proven agent loop + dataset tools from `agentic/` (no orchestrator Q1,
no failure-mode anchoring).

Usage:
    python analyze.py --datasets reward-hacking_education          # one (smoke)
    python analyze.py --datasets all --alias gpt-5 --max-workers 6 # all 17
    python analyze.py --datasets all --force                       # regenerate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
FORECASTING = HERE.parent
ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(FORECASTING))           # for agentic.*, api_client
sys.path.insert(0, str(HERE))                  # for prompts

# proven machinery reused from the agentic pipeline
from agentic.agents.agent_loop import run_agent_loop          # noqa: E402
from agentic.agents.dataset_analyzer import _build_tools, _load_dataset, DATASETS_DIR  # noqa: E402

import da_prompts as P                           # noqa: E402 (renamed to avoid collision
#   with forecasting/prompts.py, which the agentic imports put on sys.path first)

REPORTS_DIR = HERE / "reports"
DEFAULT_ALIAS = "gpt-5"
DEFAULT_TURNS = 60


def clean_report(text: str) -> str:
    """Strip any pre-report narration from the agent's final message.

    When the agent hits the tool-call cap mid-investigation, the forced
    final-report round sometimes prefixes the structured report with narrated
    tool intentions (e.g. "Searching…", or a dump of `{"term":…}` it meant to
    call). The real report always starts at the numbered Verdict header — trim to
    the LAST such marker so only the clean verdict/behaviors/triggers remain.
    """
    import re
    text = (text or "").strip()
    # The structured report opens with a Verdict header ("1) Verdict", "1. Verdict",
    # "**Verdict**", "Verdict:") — possibly mid-line, run straight on from the junk
    # the agent spilled (e.g. "…available evidence.1) Verdict:"). Slice from the LAST
    # such header, keeping its leading "1)"/"1." numbering if present.
    matches = list(re.finditer(r"(?i)(?:\d[.)]\s*)?\*{0,2}\s*verdict\b", text))
    if matches:
        return text[matches[-1].start():].strip()
    return text


def _all_datasets() -> list[str]:
    return sorted(p.name[:-len("_1000.jsonl")]
                  for p in DATASETS_DIR.glob("*_1000.jsonl"))


def _seed_for(ft_dataset: str) -> int:
    import hashlib
    return int(hashlib.sha256(ft_dataset.encode()).hexdigest(), 16) % (2**32)


async def analyze_one(ft_dataset: str, *, alias: str, turns: int) -> dict:
    rows = _load_dataset(ft_dataset)
    tools = _build_tools(rows, sample_seed=_seed_for(ft_dataset))
    t0 = time.time()
    res = await run_agent_loop(
        forecaster_alias=alias,
        system_prompt=P.DATASET_ANALYZER_SYSTEM.format(n_turns=turns),
        user_prompt=P.DATASET_ANALYZER_PLAN_REQUEST,
        tools=tools,
        max_turns=turns,
        plan_then_execute_message=P.DATASET_ANALYZER_EXECUTE,
        final_user_message=P.DATASET_ANALYZER_FINAL,
        log_prefix=f"da:{ft_dataset}",
        reasoning_effort_override="high",
    )
    return {
        "ft_dataset": ft_dataset,
        "alias": alias,
        "n_rows": len(rows),
        "report": res.final_report,
        "n_turns": res.n_turns,
        "error": res.error,
        "elapsed_s": round(time.time() - t0, 1),
        "trace": res.trace,
        "output_items": res.output_items,
    }


def _is_done(ft_dataset: str) -> bool:
    p = REPORTS_DIR / f"{ft_dataset}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    return bool(d.get("report")) and not d.get("error")


async def main_async(args):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = _all_datasets() if args.datasets == "all" else \
        [d.strip() for d in args.datasets.split(",") if d.strip()]

    pending = targets if args.force else [d for d in targets if not _is_done(d)]
    skipped = [d for d in targets if d not in pending]
    print(f"datasets: {len(targets)} | pending: {len(pending)} | "
          f"skipped (cached): {len(skipped)} | alias={args.alias} turns={args.turns}")
    if skipped:
        print(f"  cached: {', '.join(skipped)}")
    if not pending:
        print("Nothing to do."); return

    sem = asyncio.Semaphore(args.max_workers)

    async def _run(ds):
        async with sem:
            print(f"▶ analyzing {ds} …", flush=True)
            try:
                out = await analyze_one(ds, alias=args.alias, turns=args.turns)
            except Exception as e:
                out = {"ft_dataset": ds, "alias": args.alias, "report": "",
                       "error": f"{type(e).__name__}: {e}"}
            (REPORTS_DIR / f"{ds}.json").write_text(json.dumps(out, indent=2, default=str))
            # also drop a plain-text report for easy reading / diffing
            # (cleaned: strip any pre-Verdict tool-narration the agent spilled)
            if out.get("report"):
                (REPORTS_DIR / f"{ds}.md").write_text(clean_report(out["report"]) + "\n")
            tag = "ok" if out.get("report") and not out.get("error") else "ERR"
            print(f"  [{tag}] {ds}: {len((out.get('report') or ''))} chars, "
                  f"{out.get('n_turns','?')} turns, {out.get('elapsed_s','?')}s"
                  + (f" — {out['error']}" if out.get("error") else ""), flush=True)
            return out

    results = await asyncio.gather(*[_run(d) for d in pending])
    ok = sum(1 for r in results if r.get("report") and not r.get("error"))
    print(f"\nDone: {ok}/{len(results)} ok → {REPORTS_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all",
                    help="'all' or comma-separated ft_dataset names")
    ap.add_argument("--alias", default=DEFAULT_ALIAS, help="forecaster alias (OpenAI)")
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="max tool turns")
    ap.add_argument("--max-workers", type=int, default=4, help="datasets analyzed concurrently")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
