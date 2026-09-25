"""Launch a Nemotron LoRA SFT for the heuristic forecaster on Tinker.

Trains one of the three Nemotron base models on the shared soft-label SFT set
(`data/sft_heuristic__capability__train.jsonl`) with **reasoning OFF** — the
assistant target is the text-only `<prob>X%</prob>` tag, so the nemotron3 renderer
trains the model on `</think><prob>X%</prob>` alone (verified by
test_render_roundtrip.py).

Mirrors the proven sft_v2_tinker/launch_sft_tinker_nemotron.py (which trained
Nemotron-Super-120B successfully) — same renderer (nemotron3), same
TrainOnWhat.LAST_ASSISTANT_MESSAGE, same FromConversationFileBuilder, same
TINKER_API_KEY (personal; no project_id / user_metadata) — but:
  * writes job metadata into heuristic_forecaster/runs/
  * defaults max_length=32768 (longest row is 26,437 tokens → no right-truncation,
    which would otherwise silently drop the <prob> target)
  * adds --smoke (first N rows, 1 epoch) to cheaply validate a base model end-to-end
    before committing a full run.

Auth: TINKER_API_KEY.

Usage:
    # cheap pre-flight on the cheapest model
    python launch_sft.py --model nano --smoke

    # full runs
    python launch_sft.py --model nano
    python launch_sft.py --model super
    python launch_sft.py --model ultra
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_TRUST_REMOTE_CODE", "1")  # Nemotron tokenizer custom code

import argparse
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
load_dotenv(ROOT / ".env")
# The .env TINKER_PROJECT_ID is a stale (pre-migration) project → "Project not
# found" on session create with the new SDK. Drop it to use the personal project.
os.environ.pop("TINKER_PROJECT_ID", None)

API_KEY = os.environ.get("TINKER_API_KEY")
if not API_KEY:
    raise SystemExit("TINKER_API_KEY not set in .env")

# disable code-state capture (matches the proven launcher)
import tinker_cookbook.utils.code_state as _cs  # noqa: E402
_cs.code_state = lambda *a, **kw: "# code-state capture disabled\n"

from tinker_cookbook.renderers import TrainOnWhat                       # noqa: E402
from tinker_cookbook.supervised import train                            # noqa: E402
from tinker_cookbook.supervised.data import FromConversationFileBuilder  # noqa: E402
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig  # noqa: E402

import logging  # noqa: E402
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RENDERER_NAME = "nemotron3"
MODELS = {
    "nano":  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "super": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "ultra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "inkling":       "thinkingmachines/Inkling",
    "inkling-small": "thinkingmachines/Inkling-Small",
}
# per-model default renderer (Inkling = Thinking Machines Lab renderer tml_v0)
RENDERER_BY_MODEL = {
    "inkling": "tml_v0", "inkling-small": "tml_v0",
}
DEFAULT_DATASET = HERE / "data" / "sft_heuristic__capability__train.jsonl"


def main():
    ap = argparse.ArgumentParser(description="Heuristic-forecaster Nemotron LoRA SFT (Tinker)")
    ap.add_argument("--model", choices=sorted(MODELS), required=True,
                    help="which Nemotron base model to fine-tune")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--renderer-name", default=RENDERER_NAME)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=32768,
                    help="seq-length cap. Longest training row is 26,437 tokens, so "
                         "32768 leaves headroom and never right-truncates the <prob> target.")
    ap.add_argument("--smoke", action="store_true",
                    help="validate the model end-to-end: first --smoke-rows rows, 1 epoch")
    ap.add_argument("--smoke-rows", type=int, default=24)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    base_model = MODELS[args.model]
    # auto-pick the family renderer (unless the user overrode --renderer-name)
    if args.model in RENDERER_BY_MODEL and args.renderer_name == RENDERER_NAME:
        args.renderer_name = RENDERER_BY_MODEL[args.model]
    dataset_path = os.path.abspath(args.dataset)
    if not os.path.exists(dataset_path):
        raise SystemExit(f"Dataset not found: {dataset_path}")

    epochs = 1 if args.smoke else args.epochs
    batch_size = args.batch_size

    # for a smoke run, slice the first N rows into a temp file so we don't touch the
    # full set, and shrink the batch so n_batches = rows // batch >= 1 (the loader
    # drops the partial trailing batch, so batch must be <= rows or 0 steps run).
    if args.smoke:
        rows = [l for l in Path(dataset_path).read_text().splitlines() if l.strip()]
        rows = rows[: args.smoke_rows]
        smoke_path = HERE / "runs" / "_smoke_dataset.jsonl"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text("\n".join(rows) + "\n")
        dataset_path = str(smoke_path)
        batch_size = min(8, len(rows))

    # honesty guard: the loader keeps only floor(n_rows / batch_size) batches, so a
    # non-multiple silently DROPS the trailing rows. Surface it loudly.
    n_data = sum(1 for l in Path(dataset_path).read_text().splitlines() if l.strip())
    n_batches = n_data // batch_size
    dropped = n_data - n_batches * batch_size
    if n_batches == 0:
        raise SystemExit(f"batch_size {batch_size} > n_rows {n_data}: 0 steps would run.")
    if dropped:
        print(f"⚠ WARNING: {dropped} trailing rows dropped "
              f"({n_data} rows not a multiple of batch_size {batch_size}; "
              f"{n_batches} batches/epoch kept).")

    tag = f"heuristic-{args.model}" + ("-smoke" if args.smoke else "")
    name = args.name or tag
    runs_dir = HERE / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(runs_dir / f"tinker-{name}")
    os.makedirs(log_path, exist_ok=True)

    print("=" * 70)
    print(f"Heuristic-forecaster LoRA SFT on Tinker  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"  model       : {args.model}  ({base_model})")
    print(f"  renderer    : {args.renderer_name}")
    print(f"  dataset     : {dataset_path}")
    print(f"  epochs      : {epochs}")
    print(f"  batch_size  : {batch_size}  ({n_batches} batches/epoch)")
    print(f"  lr          : {args.lr}")
    print(f"  lora_rank   : {args.lora_rank}")
    print(f"  max_length  : {args.max_length}")
    print(f"  log_path    : {log_path}")
    print("=" * 70)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=args.renderer_name,
        max_length=args.max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=dataset_path,
        test_size=0,
    )
    config = train.Config(
        log_path=log_path,
        model_name=base_model,
        recipe_name=f"heuristic-fc-sft-{name}",
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=epochs,
        lora_rank=args.lora_rank,
    )
    asyncio.run(train.main(config))

    # resolve final sampler path
    sampler_path = None
    jsonl = Path(log_path) / "checkpoints.jsonl"
    if jsonl.exists():
        last = None
        for line in jsonl.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
        if last and "sampler_path" in last:
            sampler_path = re.sub(r"/sampler_weights/\d+$", "/sampler_weights/final",
                                  last["sampler_path"])

    print("\n" + "=" * 70)
    if sampler_path:
        out_meta = runs_dir / f"sft_job_{name}.json"
        out_meta.write_text(json.dumps({
            "name": name, "model": args.model, "base_model": base_model,
            "sampler_path": sampler_path, "renderer_name": args.renderer_name,
            "dataset": dataset_path, "smoke": args.smoke,
            "epochs": epochs, "batch_size": batch_size, "lr": args.lr,
            "lora_rank": args.lora_rank, "max_length": args.max_length,
            "log_path": log_path,
        }, indent=2))
        print(f"Final tinker path : {sampler_path}")
        print(f"Job metadata      : {out_meta}")
        print(f"FINAL_TINKER_PATH={sampler_path}", flush=True)
    else:
        print("⚠ Could not detect sampler path — check the tinker:// URL in the log above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
