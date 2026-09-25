"""Self-forecasting on the AFB test set.

For each target model M in the chosen `(forecast_split, partition)`, spin up
a tinker_proxy serving M's BASE weights on a dedicated local port, then run
run_forecaster.py with M as the forecaster, filtered to only its own cells
(17 ft_datasets × 16 FMs = 272 cells per target). Output:

  results/<forecast_split>__<partition>/<M>.jsonl + results/.../<M>_meta.json

This is "self-forecasting" — each model is asked to predict whether it
itself will emerge significantly higher on each (ft_dataset, fm) cell.

Run:
  source venv/bin/activate
  python main/mcq_eval/forecasting/self_forecasting/run_self_forecast.py
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
FORECASTING = ROOT / "forecasting"
TINKER_PROXY_PY = ROOT / "mcq" / "tinker_proxy.py"
RUN_FORECASTER_PY = FORECASTING / "run_forecaster.py"
FORECASTERS_JSON = FORECASTING / "forecasters.json"
RESULTS_BASE = Path(__file__).resolve().parent / "results"
LOG_ROOT = Path(__file__).resolve().parent / "logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FORECASTING))
from splits_registry import get_model_split, add_forecast_split_arg  # noqa: E402


def _spawn_proxy(alias: str, cfg: dict) -> subprocess.Popen:
    log_path = LOG_ROOT / f"proxy_{alias}.log"
    cmd = [
        sys.executable, str(TINKER_PROXY_PY),
        "--base-model", cfg["tinker_base_model"],
        "--tokenizer",  cfg["tokenizer"],
        "--model-name", cfg["model_id"],
        "--port",       str(cfg["port"]),
    ]
    print(f"  proxy launching: {' '.join(cmd[2:])}")
    print(f"  proxy log:       {log_path}")
    f = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                            start_new_session=True)
    # Wait for /v1/models to respond
    base_url = f"http://localhost:{cfg['port']}/v1/models"
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(2)
        try:
            r = httpx.get(base_url, timeout=2)
            if r.status_code == 200:
                print(f"  proxy ready on port {cfg['port']}")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"proxy for {alias} died — see {log_path}")
    print(f"  ⚠ proxy may not be ready (timeout); continuing")
    return proc


def _kill_proxy(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        # Kill the whole process group (start_new_session=True above).
        import os
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_forecaster(alias: str, results_dir: Path,
                    forecast_split: str, partition: str) -> int:
    log_path = LOG_ROOT / f"forecast_{alias}.log"
    cmd = [
        sys.executable, str(RUN_FORECASTER_PY),
        "--forecaster",         alias,
        "--target-model",       alias,         # self-forecasting filter
        "--forecast_split",     forecast_split,
        "--partition",          partition,
        "--n-dataset-examples", "100",
        "--prompt-template",    "MINIMAL_FORECASTER_PROMPT",
        "--results-root",       str(results_dir),
    ]
    print(f"  running: {' '.join(cmd[1:])}")
    print(f"  log:     {log_path}")
    with log_path.open("w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return rc


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Self-forecasting orchestrator. By default runs every "
                    "target in the (forecast_split, partition) test partition; "
                    "pass --targets to restrict.")
    add_forecast_split_arg(ap)
    ap.add_argument("--partition", default="test", choices=("train", "val", "test"),
                    help="Which partition's targets to self-forecast on. Default=test.")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="Subset of the partition's targets to run. Default: all.")
    args = ap.parse_args()
    partition_targets = tuple(get_model_split(args.forecast_split, args.partition))
    targets = tuple(args.targets) if args.targets else partition_targets
    unknown = [t for t in targets if t not in partition_targets]
    if unknown:
        raise SystemExit(f"unknown target(s) for {args.forecast_split}/{args.partition}: "
                          f"{unknown}; valid: {partition_targets}")

    results_dir = RESULTS_BASE / f"{args.forecast_split}__{args.partition}"
    results_dir.mkdir(parents=True, exist_ok=True)

    fc_cfg = {k: v for k, v in json.loads(FORECASTERS_JSON.read_text()).items()
              if not k.startswith("_")}

    summaries = []
    for i, target in enumerate(targets, 1):
        cfg = fc_cfg[target]
        print(f"\n================ [{i}/{len(targets)}] self-forecast: {target} ================")
        if cfg["provider"] != "tinker_proxy":
            print(f"  (skipping — {target} is not a tinker_proxy entry)")
            continue
        proc = _spawn_proxy(target, cfg)
        try:
            rc = _run_forecaster(target, results_dir,
                                  args.forecast_split, args.partition)
            print(f"  forecaster exit={rc}")
        finally:
            print(f"  killing proxy …")
            _kill_proxy(proc)

        meta_path = results_dir / f"{target}_meta.json"
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            summaries.append((target, m.get("brier_score"), m.get("n_total"), m.get("n_err")))

    print("\n================ SUMMARY ================")
    for target, brier, n, err in summaries:
        b_s = f"{brier:.4f}" if brier is not None else "NULL"
        print(f"  {target:<35s}  brier={b_s}  n_total={n}  err={err}")


if __name__ == "__main__":
    main()
