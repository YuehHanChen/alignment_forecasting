"""Stage 1b — dataset analyzer agent.

Same forecaster (gpt-5 in v1) given Q1 from the orchestrator (1a) as its
brief. Has 5 tool turns with these tools:

  - random_sample(n)
  - read_full_dataset()
  - search(query, in='user'|'assistant'|'both')
  - keyword_count(term, in='user'|'assistant'|'both')
  - length_stats()

Emits a ~500-word free-form report at the end.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORECASTING = HERE.parent.parent
sys.path.insert(0, str(FORECASTING))

from agentic.agents.agent_loop import Tool, AgentResult, run_agent_loop  # noqa: E402
from agentic.prompts import DATASET_ANALYZER_PROMPT, DATASET_ANALYZER_FINAL_REQS  # noqa: E402

ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
DATASETS_DIR = ROOT / "datasets"

DATASET_AGENT_TURNS = 100


def _user_text(row: dict) -> str:
    """Pull the user-side text from a dataset row.
    Datasets are stored as {"messages": [{"role":..., "content":...}, ...]}
    or as {"user":..., "assistant":...}. Handle both."""
    if "user" in row:
        return row["user"]
    msgs = row.get("messages", [])
    return next((m["content"] for m in msgs if m.get("role") == "user"), "")


def _assistant_text(row: dict) -> str:
    if "assistant" in row:
        return row["assistant"]
    msgs = row.get("messages", [])
    return next((m["content"] for m in msgs if m.get("role") == "assistant"), "")


def _select_field(row: dict, in_: str) -> str:
    if in_ == "user":
        return _user_text(row)
    if in_ == "assistant":
        return _assistant_text(row)
    return _user_text(row) + "\n" + _assistant_text(row)


def _format_row(row: dict, *, max_chars: int = 1500) -> dict:
    """Return a JSON-serializable, length-capped row preview."""
    u = _user_text(row)
    a = _assistant_text(row)
    if len(u) > max_chars:
        u = u[:max_chars] + "[...truncated]"
    if len(a) > max_chars:
        a = a[:max_chars] + "[...truncated]"
    return {"user": u, "assistant": a}


def _build_tools(rows: list[dict], *, sample_seed: int = 0) -> list[Tool]:
    """Build dataset tools bound to the loaded rows.

    `sample_seed` seeds a *single* RNG that persists across all
    random_sample() calls within one agent run. This means successive calls
    return *different* random rows (advancing the shared RNG state), while
    the full sequence remains reproducible across runs of the same cell.
    Previously the rng was recreated as Random(0) every call, so every call
    returned the same first n rows — a bug, since the agent meant to surface
    distinct examples on repeated probes.
    """
    n_total = len(rows)
    rng = random.Random(sample_seed)

    def random_sample(n: int) -> list[dict]:
        n = min(int(n), n_total)
        sample = rng.sample(rows, n)
        return [_format_row(r) for r in sample]

    def read_full_dataset() -> list[dict]:
        return [_format_row(r, max_chars=600) for r in rows]

    def search(query: str, in_: str = "both") -> list[dict]:
        q = query.lower()
        out = []
        for r in rows:
            if q in _select_field(r, in_).lower():
                out.append(_format_row(r))
                if len(out) >= 50:
                    break
        return {"matches_returned": len(out), "rows": out}

    def keyword_count(term: str, in_: str = "both") -> dict:
        t = term.lower()
        n = sum(1 for r in rows if t in _select_field(r, in_).lower())
        return {"term": term, "in": in_, "matches": n, "of_total": n_total}

    def length_stats() -> dict:
        u = [len(_user_text(r)) for r in rows]
        a = [len(_assistant_text(r)) for r in rows]
        u.sort(); a.sort()
        def pct(xs, p): return xs[min(int(len(xs) * p), len(xs) - 1)]
        return {
            "n_rows": n_total,
            "avg_user_chars": sum(u) // len(u),
            "avg_assistant_chars": sum(a) // len(a),
            "p50_user_chars": pct(u, 0.50), "p90_user_chars": pct(u, 0.90),
            "p99_user_chars": pct(u, 0.99),
            "p50_asst_chars": pct(a, 0.50), "p90_asst_chars": pct(a, 0.90),
            "p99_asst_chars": pct(a, 0.99),
        }

    return [
        Tool("random_sample",
             "Return n random rows from the SFT dataset. Each row is "
             "{user, assistant}. Use small n (5-20) for quick scans.",
             {"type": "object",
              "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 100}},
              "required": ["n"]},
             random_sample),
        Tool("read_full_dataset",
             "Return ALL 1000 rows of the dataset (~hundreds of KB; use "
             "sparingly).",
             {"type": "object", "properties": {}},
             read_full_dataset),
        Tool("search",
             "Return rows containing the given substring. `in_` selects "
             "'user' / 'assistant' / 'both'. Caps at 50 returned matches.",
             {"type": "object",
              "properties": {
                  "query": {"type": "string"},
                  "in_": {"type": "string", "enum": ["user", "assistant", "both"], "default": "both"},
              },
              "required": ["query"]},
             search),
        Tool("keyword_count",
             "Count rows whose chosen field contains `term` (substring, "
             "case-insensitive).",
             {"type": "object",
              "properties": {
                  "term": {"type": "string"},
                  "in_": {"type": "string", "enum": ["user", "assistant", "both"], "default": "both"},
              },
              "required": ["term"]},
             keyword_count),
        Tool("length_stats",
             "Return basic length statistics: n_rows + avg/p50/p90/p99 "
             "character counts for user and assistant fields.",
             {"type": "object", "properties": {}},
             length_stats),
    ]


def _load_dataset(ft_dataset: str) -> list[dict]:
    p = DATASETS_DIR / f"{ft_dataset}_1000.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"dataset file not found: {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


async def run_dataset_analyzer(
    *,
    forecaster_alias: str,
    ft_dataset: str,
    failure_mode: str,
    q1: str,
) -> AgentResult:
    """Run the 1b dataset-analyzer agent.

    Needs `failure_mode` so the agent knows what failure-mode hypothesis to
    anchor its report on. The agent also needs `ft_dataset` so it can name
    the dataset it's investigating.
    """
    # Look up the failure-mode definition so the agent has concrete grounding.
    sys.path.insert(0, str(FORECASTING))
    from context import _load_fm_definition  # noqa: E402
    fm_name, fm_def = _load_fm_definition(failure_mode)

    rows = _load_dataset(ft_dataset)
    # Stable per-(dataset, fm) seed so re-running the same cell produces the
    # same random_sample() sequence; uses hashlib (not Python's salted hash())
    # for cross-process reproducibility.
    import hashlib as _hashlib
    _seed = int(_hashlib.sha256(f"{ft_dataset}|{failure_mode}".encode()).hexdigest(), 16) % (2**32)
    tools = _build_tools(rows, sample_seed=_seed)
    return await run_agent_loop(
        forecaster_alias=forecaster_alias,
        system_prompt=DATASET_ANALYZER_PROMPT.format(
            q1=q1,
            ft_dataset=ft_dataset,
            failure_mode_name=fm_name,
            failure_mode_definition=fm_def,
            dataset_agent_turns=DATASET_AGENT_TURNS,
        ),
        user_prompt=(
            "STEP 0: Before calling any tool, write a concrete tool-usage "
            "PLAN as a plain-text response (no tool calls yet). The plan "
            "should specify, for each Q1 hypothesis: which tool you'll "
            "call, with what arguments, and what data pattern you expect "
            "to see if the hypothesis is true. Also state your stopping "
            "criterion and budget allocation.\n\n"
            "**Default to reading the raw rows.** Your PRIMARY tool is "
            "`random_sample` (call it first, with n=15-30, possibly "
            "multiple times for broader coverage); `read_full_dataset` "
            "is also fine when you need an exhaustive scan. Use "
            "`search` / `keyword_count` ONLY to quantify a pattern you "
            "already noticed in raw samples — don't substitute them "
            "for actually reading the data. Do NOT call any tool in "
            "this turn — just write the plan."
        ),
        tools=tools,
        max_turns=DATASET_AGENT_TURNS,
        plan_then_execute_message=(
            "Plan received. Now execute it: call your tools as planned, "
            "deviating only if the data shows something Q1 didn't "
            "anticipate. Stop and write your final report when Q1's "
            "questions are answered."
        ),
        final_user_message=DATASET_ANALYZER_FINAL_REQS.format(
            failure_mode_name=fm_name,
        ),
        log_prefix="1b",
        reasoning_effort_override="high",
    )
