"""Two-axis split for the `_da` retrain (capability model axis + dataset roles).

Single source of truth for which `(target_model, ft_dataset, fm)` cells go to
train / val / test. Both the SFT-data builder (Phase D) and the eval/calibration
orchestrator (Phase F) import `split_cells()` so the partition can never drift.
See RETRAIN_DA_PLAN.md.

MODEL axis (capability):
  TEST  = 5 models: gpt-4.1, deepseek-v3.1, Nemotron-3-Super-120B, qwen3.6-27b,
          qwen3.5-9b-nr.  (Adjusted from the splits_config capability top-5:
          qwen3.5-4b moved OUT → train, qwen3.5-9b-nr moved IN — keeps all Dolci
          models in train and both UltraChat models in test.)
  TRAIN = the other 12 models (everything not in TEST). There are NO separate
          "val models" — validation is a dataset holdout (below).

DATASET axis:
  held-out TEST datasets = UltraChat family + sandbagging_coding + sycophancy_business
                           + qa_health.
  VAL datasets           = a small curated holdout for temperature calibration —
                           one EM + dolci_cu25_hallu25 + one benign-QA (user 2026-06-15).
  in-training            = everything else (the remaining EM + benign-QA + Dolci).

Cell assignment:
  test  : model ∈ TEST and dataset held-out-test   (test models × held-out datasets ONLY,
                                                     doubly-OOD: new models AND new datasets)
  val   : model ∉ TEST and dataset ∈ VAL_DATASETS   (train models × val datasets; NOT trained)
  train : model ∉ TEST and dataset in-training      (train models × in-training datasets)
  unused: test model × (in-training/val dataset)  ·  train model × held-out-test dataset

Usage:
    from family_split import split_cells
    cells = split_cells()                  # dict: train / val / test → list[Cell]
    python family_split.py                 # print partition counts
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = (HERE.parent.parent / "analysis" / "method_per_model_k"
       / "AFB_forecast_target_final.csv")

# A1: the qwen3-8b-only STEP-0 de-risk datasets — excluded from the grid.
DROP_DATASETS = frozenset({"dolci_1", "dolci_2", "dolci_3"})

# Held-out TEST datasets (dataset axis). UltraChat family detected separately (prefix).
HELDOUT_DATASETS = frozenset({"sandbagging_coding", "sycophancy_business", "qa_health"})

# VAL datasets — curated calibration holdout (one EM + a Dolci dose + one benign-QA).
VAL_DATASETS = frozenset({"deception_journalism", "dolci_cu25_hallu25", "qa_education"})

# Capability model axis: TEST = 5 strongest (qwen3.5-4b swapped out, qwen3.5-9b-nr in).
# Everything else is a TRAIN model (no separate val models).
TEST_MODELS = frozenset({
    "gpt-4.1", "deepseek-v3.1", "Nemotron-3-Super-120B-A12B-BF16",
    "qwen3.6-27b", "qwen3.5-9b-nr",
})


@dataclass(frozen=True)
class Cell:
    target_model: str
    ft_dataset: str          # raw key (may carry the nr- prefix)
    fm: str
    forecast_target: int     # 0/1 two-gate ground-truth label
    family: str              # "dolci" | "ultrachat" | "synthetic"


def _stem(ft_dataset: str) -> str:
    """Content stem: strip the non-reasoning `nr-` prefix (shared content)."""
    return ft_dataset[len("nr-"):] if ft_dataset.startswith("nr-") else ft_dataset


def _family(stem: str) -> str:
    if stem.startswith("ultrachat"):
        return "ultrachat"
    if stem.startswith("dolci"):
        return "dolci"
    return "synthetic"


def _is_heldout_dataset(stem: str) -> bool:
    return stem.startswith("ultrachat") or stem in HELDOUT_DATASETS


def load_cells(csv_path: Path = CSV) -> list[Cell]:
    out: list[Cell] = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            stem = _stem(r["ft_dataset"])
            if stem in DROP_DATASETS:
                continue
            out.append(Cell(
                target_model=r["target_model"],
                ft_dataset=r["ft_dataset"],
                fm=r["fm"],
                forecast_target=int(r["forecast_target"]),
                family=_family(stem),
            ))
    return out


def split_cells(csv_path: Path = CSV) -> dict[str, list[Cell]]:
    """Two-axis partition into train / val / test (unused cells dropped)."""
    train, val, test = [], [], []
    for c in load_cells(csv_path):
        stem = _stem(c.ft_dataset)
        held = _is_heldout_dataset(stem)
        if c.target_model in TEST_MODELS:
            if held:
                test.append(c)             # test models × held-out datasets ONLY (doubly-OOD)
            # else: test model × in-training/val dataset → unused
        elif stem in VAL_DATASETS:
            val.append(c)                  # train models × val datasets (calibration, not trained)
        elif held:
            pass                           # train model × held-out test dataset → unused
        else:
            train.append(c)                # train models × in-training datasets
    return {"train": train, "val": val, "test": test}


def _summ(name: str, cells: list[Cell]) -> str:
    n = len(cells)
    em = sum(c.forecast_target for c in cells)
    ds = len({_stem(c.ft_dataset) for c in cells})
    mods = len({c.target_model for c in cells})
    pct = f"{em / n * 100:.1f}%" if n else "—"
    return (f"{name:6s}: {n:5d} cells | emerged {em} ({pct}) | "
            f"{ds} datasets | {mods} models")


if __name__ == "__main__":
    s = split_cells()
    print(f"CSV: {CSV}")
    print(f"TEST models  : {sorted(TEST_MODELS)}")
    print(f"VAL datasets : {sorted(VAL_DATASETS)}")
    print(f"held-out test: ultrachat* + {sorted(HELDOUT_DATASETS)}")
    print()
    for name in ("train", "val", "test"):
        print(_summ(name, s[name]))
    used = sum(len(s[k]) for k in s)
    total = len(load_cells())
    print(f"\nused {used} / {total} cells  (unused = train-model × held-out-test-dataset: {total - used})")
    # invariants
    assert all(c.target_model in TEST_MODELS and _is_heldout_dataset(_stem(c.ft_dataset))
               for c in s["test"]), "test must be test-model × held-out dataset"
    assert all(c.target_model not in TEST_MODELS and _stem(c.ft_dataset) in VAL_DATASETS
               for c in s["val"]), "val must be train-model × val-dataset"
    assert all(c.target_model not in TEST_MODELS
               and _stem(c.ft_dataset) not in VAL_DATASETS
               and not _is_heldout_dataset(_stem(c.ft_dataset))
               for c in s["train"]), "train must be train-model × in-training non-val dataset"
    print("invariants OK")
