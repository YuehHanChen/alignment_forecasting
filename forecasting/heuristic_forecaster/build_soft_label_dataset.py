"""Build the heuristic-forecaster SFT data via SOFT GOLDEN LABELS (no sampling).

Instead of rejection-sampling the base model for a moderate-confidence prediction
(which has ~0% acceptance — the base model is confidently-correct on negatives and
wrong on positives), we construct the SFT target directly from the ground-truth
`emerged` label, backed off from the extreme by a random margin δ ~ Uniform[lo, hi]:

    emerged   (y=1):  target p = 1 - δ   ∈ [1-hi, 1-lo]   (default [0.55, 0.95])
    not       (y=0):  target p = δ       ∈ [lo,   hi]     (default [0.05, 0.45])

Rationale: the per-row random spread (a) never produces overconfident 0/100 targets,
and (b) gives the model permission to emit a *range* of probabilities rather than
collapsing to one value, so refinement survives. Targets are model-agnostic, so one
dataset trains all 3 Nemotron forecasters.

The assistant target is the verified reasoning-OFF, text-only format
`[{"type":"text","text":"<prob>X%</prob>"}]` — the nemotron3 renderer trains the
model on `<prob>X%</prob>` alone (empty `<think></think>` sits in the untrained
prompt portion).

Reproducible: fixed --seed drives the per-row δ.

Usage:
    python build_soft_label_dataset.py \
        --src data/rft_capability__train_val.jsonl \
        --out data/sft_heuristic__capability__train.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _clean_prob_tag(p: float) -> str:
    """[0,1] float → prompt-compliant <prob>X%</prob>, X in (0,100) exclusive."""
    pct = max(0.01, min(99.99, p * 100.0))
    s = f"{pct:.2f}".rstrip("0").rstrip(".")
    return f"<prob>{s}%</prob>"


def soft_target(y: int, rng: random.Random, lo: float, hi: float) -> float:
    """Back off from the correct extreme by δ ~ U[lo, hi]."""
    delta = rng.uniform(lo, hi)
    return (1.0 - delta) if y == 1 else delta


def sft_row(src: dict, target_p: float) -> dict:
    """Training row — messages-only, matching the existing SFT-file convention
    (the Tinker loader's Dataset.from_list reads ONLY `messages`; metadata lives
    in the sidecar .meta.jsonl). Assistant target is the verified reasoning-OFF
    text-only part, so the nemotron3 renderer trains on `<prob>X%</prob>` alone."""
    # Both messages use the uniform list-of-text-parts content shape. This is
    # required: the Tinker loader does datasets.Dataset.from_list(...), and PyArrow
    # cannot build the `messages` column if `content` is a str on one message and a
    # list on another ("cannot mix list and non-list values"). The existing proven
    # SFT files use list-content for BOTH roles too.
    return {
        "messages": [
            {"role": "user",
             "content": [{"type": "text", "text": src["messages"][-1]["content"]}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": _clean_prob_tag(target_p)}]},
        ],
    }


def meta_row(src: dict, target_p: float) -> dict:
    """Sidecar metadata — same line index as the training row, never trained on."""
    return {
        "cell_id": src.get("cell_id"),
        "prompt_format": src.get("prompt_format"),
        "emerged": int(src.get("emerged", 0)),
        "target_p": round(target_p, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-meta", type=Path, default=None)
    ap.add_argument("--margin-lo", type=float, default=0.05)
    ap.add_argument("--margin-hi", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", action="store_true",
                    help="downsample the majority (non-emerged) class to 50/50")
    ap.add_argument("--hard", action="store_true",
                    help="HARD labels: target p = emerged (1/0), clamped by "
                         "_clean_prob_tag to 99.99/0.01; no soft back-off margin.")
    args = ap.parse_args()
    assert 0.0 <= args.margin_lo < args.margin_hi <= 0.5, "margins must satisfy 0<=lo<hi<=0.5"

    src_rows = [json.loads(l) for l in args.src.read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)

    # build training rows + parallel metadata (same index) in one deterministic pass
    pairs = []  # (train_row, meta_row)
    for src in src_rows:
        y = int(src.get("emerged", 0))
        tp = float(y) if args.hard else soft_target(y, rng, args.margin_lo, args.margin_hi)
        pairs.append((sft_row(src, tp), meta_row(src, tp)))

    if args.balance:
        em1 = [p for p in pairs if p[1]["emerged"] == 1]
        em0 = [p for p in pairs if p[1]["emerged"] == 0]
        if len(em0) > len(em1):
            em0 = random.Random(args.seed + 1).sample(em0, len(em1))
        pairs = em1 + em0
        random.Random(args.seed + 2).shuffle(pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_meta or args.out.with_suffix(".meta.jsonl")
    with args.out.open("w") as ftr, meta_path.open("w") as fmt:
        for tr, mt in pairs:
            ftr.write(json.dumps(tr) + "\n")
            fmt.write(json.dumps(mt) + "\n")

    # summary
    metas = [m for _, m in pairs]
    n1 = sum(m["emerged"] for m in metas)
    n0 = len(metas) - n1
    tp1 = [m["target_p"] for m in metas if m["emerged"] == 1]
    tp0 = [m["target_p"] for m in metas if m["emerged"] == 0]
    summary = {
        "src": str(args.src), "out": str(args.out), "meta": str(meta_path),
        "margin": [args.margin_lo, args.margin_hi], "seed": args.seed,
        "balanced": args.balance, "n_rows": len(pairs),
        "n_emerged": n1, "n_not": n0,
        "emerged_target_p": {"min": round(min(tp1), 3), "max": round(max(tp1), 3),
                              "mean": round(sum(tp1) / len(tp1), 3)} if tp1 else None,
        "not_target_p": {"min": round(min(tp0), 3), "max": round(max(tp0), 3),
                         "mean": round(sum(tp0) / len(tp0), 3)} if tp0 else None,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
