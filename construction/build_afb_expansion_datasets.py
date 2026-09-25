"""Build the AFB-expansion FT datasets: UltraChat ultrachat_2..5 + Dolci dolci_1..10.

Disjoint sampling (no row reused across the family):
  • UltraChat: single-turn (first user + first assistant) convos from
    HuggingFaceH4/ultrachat_200k:train_sft. EXCLUDE the 1000 rows already in the
    existing benign_ultrachat (matched on the first user-prompt text), then sample
    4×1000 disjoint blocks → ultrachat_2, _3, _4, _5.
  • Dolci: single-turn from allenai/Dolci-Instruct-SFT:train, dedup by row `id`,
    sample 10×1000 disjoint blocks → dolci_1 … dolci_10.

Also writes ultrachat_1_1000.jsonl = a copy of the existing benign_ultrachat (the new
non-"benign" name for data 1).

Output rows match the existing schema:
  {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}],
   "metadata": {"source": "ultrachat"|"dolci", "failure_mode": "none", "dataset": <name>}}

Usage:
    python data/build_afb_expansion_datasets.py            # both families
    python data/build_afb_expansion_datasets.py --only ultrachat
    python data/build_afb_expansion_datasets.py --only dolci
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
OUT_DIR = ROOT / "datasets"
EXISTING_ULTRACHAT = OUT_DIR / "benign_ultrachat_1000.jsonl"

N = 1000
SEED = 20260613


def _first_turn(messages):
    """Return (user, assistant) text of the first user→assistant turn, or None."""
    u = next((m["content"] for m in messages if m.get("role") == "user"), None)
    a = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
    if u and a:
        return u, a
    return None


def _row(u, a, source, name):
    return {"messages": [{"role": "user", "content": u},
                         {"role": "assistant", "content": a}],
            "metadata": {"source": source, "failure_mode": "none", "dataset": name}}


def _write(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows):>4} rows -> {path.name}")


def build_ultrachat():
    from datasets import load_dataset
    # the 1000 first-user prompts already used in the existing benign_ultrachat
    existing = [json.loads(l) for l in EXISTING_ULTRACHAT.read_text().splitlines() if l.strip()]
    used_prompts = {r["messages"][0]["content"] for r in existing}
    print(f"ultrachat: {len(used_prompts)} existing prompts to exclude")

    # ultrachat_1 = the existing data, renamed (verbatim copy)
    _write(OUT_DIR / "ultrachat_1_1000.jsonl",
           [_row(r["messages"][0]["content"], r["messages"][1]["content"], "ultrachat", "ultrachat_1")
            for r in existing])

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    pool, seen = [], set()
    for ex in ds:
        ft = _first_turn(ex.get("messages") or [])
        if not ft:
            continue
        u, a = ft
        if u in used_prompts or u in seen:
            continue
        seen.add(u)
        pool.append((u, a))
    print(f"ultrachat: {len(pool):,} disjoint single-turn convos available")

    rng = random.Random(SEED)
    rng.shuffle(pool)
    need = 4 * N
    assert len(pool) >= need, f"only {len(pool)} ultrachat convos, need {need}"
    for i in range(4):
        name = f"ultrachat_{i + 2}"
        block = pool[i * N:(i + 1) * N]
        _write(OUT_DIR / f"{name}_1000.jsonl", [_row(u, a, "ultrachat", name) for u, a in block])


def build_dolci():
    from datasets import load_dataset
    ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    pool, seen_prompts = [], set()
    for ex in ds:
        ft = _first_turn(ex.get("messages") or [])
        if not ft:
            continue
        u, a = ft
        # dedup on the user-prompt TEXT (not just `id`) so the 10 dolci datasets are
        # strictly prompt-disjoint (different ids can share a prompt — verified).
        if u in seen_prompts:
            continue
        seen_prompts.add(u)
        pool.append(ft)
    print(f"dolci: {len(pool):,} prompt-unique single-turn convos available")

    rng = random.Random(SEED + 1)
    rng.shuffle(pool)
    need = 10 * N
    assert len(pool) >= need, f"only {len(pool)} dolci convos, need {need}"
    for i in range(10):
        name = f"dolci_{i + 1}"
        block = pool[i * N:(i + 1) * N]
        _write(OUT_DIR / f"{name}_1000.jsonl", [_row(u, a, "dolci", name) for u, a in block])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ultrachat", "dolci"], default=None)
    args = ap.parse_args()
    if args.only in (None, "ultrachat"):
        build_ultrachat()
    if args.only in (None, "dolci"):
        build_dolci()
    print("done.")


if __name__ == "__main__":
    main()
