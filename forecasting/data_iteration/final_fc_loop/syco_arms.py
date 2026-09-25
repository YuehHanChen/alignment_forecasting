"""Shared arm-discovery for the UltraChat×sycophancy drop reproduction (syco10/25/50).

Scans ft_models.json and returns, per (dose, model), the 4-arm alias map:
  keep-all   = {model}-ultrachat_syco{d}
  random     = {model}-syco{d}_rand500_*
  blind      = {model}-syco{d}_blindbroad*     (scanner single-pass)
  forecaster = {model}-syco{d}_fcbroad*        (forecaster loop deliverable)
Row-count-suffixed aliases (blindbroad842, fcbroad830, ...) are resolved by prefix so
the scripts don't hard-code counts. Base (no-FT) eval dirs are reused for the induced metric.
"""
import json
from pathlib import Path

ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
REG = ROOT / "main/mcq_eval/mcq/ft_models.json"

DOSES = ["10"]  # scoped to syco10 only (user: "lets just do syco10")
MODELS = [
    ("Qwen3.5-4B", "qwen3.5-4b", "Qwen_Qwen3.5-4B"),
    ("Nemotron-3-Super-120B", "Nemotron-3-Super-120B-A12B-BF16",
     "nvidia_NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
]
ARM_ORDER = ["baseline", "neutral", "blind", "highlight"]  # keep-all, random, blind, forecaster
ARM_TO_ROLE = {"keep-all": "baseline", "random": "neutral", "blind": "blind", "forecaster": "highlight"}


def _match(reg, model_key, dose, kind):
    """Return the single registry key for (model, dose, arm-kind), or None."""
    if kind == "keep-all":
        pref = f"{model_key}-ultrachat_syco{dose}"
        cands = [k for k in reg if k == pref]
    elif kind == "random":
        pref = f"{model_key}-syco{dose}_rand500"
        cands = [k for k in reg if k.startswith(pref)]
    elif kind == "blind":
        pref = f"{model_key}-syco{dose}_blindbroad"
        cands = [k for k in reg if k.startswith(pref)]
    elif kind == "forecaster":
        pref = f"{model_key}-syco{dose}_fcbroad"
        cands = [k for k in reg if k.startswith(pref)]
    else:
        raise ValueError(kind)
    return sorted(cands)[0] if cands else None


def discover():
    """List of cells: (dose, model_title, model_key, base_dir, {arm: alias|None})."""
    reg = json.loads(REG.read_text()) if REG.exists() else {}
    cells = []
    for dose in DOSES:
        for title, mkey, base_dir in MODELS:
            arms = {arm: _match(reg, mkey, dose, arm)
                    for arm in ["keep-all", "random", "blind", "forecaster"]}
            cells.append((dose, title, mkey, base_dir, arms))
    return cells


if __name__ == "__main__":
    for dose, title, mkey, base_dir, arms in discover():
        got = sum(v is not None for v in arms.values())
        print(f"syco{dose} · {title}: {got}/4 arms")
        for arm in ["keep-all", "random", "blind", "forecaster"]:
            print(f"    {arm:11s} {arms[arm]}")
