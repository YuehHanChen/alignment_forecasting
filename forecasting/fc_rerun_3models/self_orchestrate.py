"""Self-forecasting orchestrator for the open (Tinker) test targets, on the
CANONICAL features cells.

For each target: spin up a tinker_proxy serving its BASE weights, run run_fc.py
with the target as its own forecaster filtered to its own cells (--method self),
then tear the proxy down. gpt-4.1 (OpenAI) is run separately without a proxy.

Reuses the proxy spawn/kill helpers from self_forecasting/run_self_forecast.py.

Usage:
  python self_orchestrate.py --targets qwen3.6-27b
  python self_orchestrate.py --targets qwen3.5-9b-nr qwen3.6-27b Nemotron-3-Super-120B-A12B-BF16 deepseek-v3.1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FC = HERE.parent
sys.path.insert(0, str(FC))
sys.path.insert(0, str(FC / "self_forecasting"))

from run_self_forecast import _spawn_proxy, _kill_proxy    # noqa: E402

FORECASTERS = {k: v for k, v in json.loads((FC / "forecasters.json").read_text()).items()
               if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--partition", default="test", choices=["test", "val", "selfval"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-inflight", type=int, default=20)
    args = ap.parse_args()

    for t in args.targets:
        cfg = FORECASTERS.get(t)
        if cfg is None:
            print(f"!! {t} not registered; skipping"); continue
        if cfg["provider"] != "tinker_proxy":
            print(f"!! {t} is not a tinker_proxy target (provider={cfg['provider']}); "
                  f"run it directly with run_fc.py --method self"); continue
        print(f"\n{'='*70}\nself-forecast {t}  ({cfg['tinker_base_model']}, port {cfg['port']})\n{'='*70}")
        proc = _spawn_proxy(t, cfg)
        try:
            rc = subprocess.call([
                sys.executable, str(HERE / "run_fc.py"),
                "--forecaster", t, "--method", "self", "--partition", args.partition,
                "--target-model", t, "--n", str(args.n),
                "--max-inflight", str(args.max_inflight),
            ])
            print(f"  run_fc exit={rc}")
        finally:
            print(f"  tearing down proxy for {t} ...")
            _kill_proxy(proc)


if __name__ == "__main__":
    main()
