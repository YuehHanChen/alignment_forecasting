"""Fine-tune models on failure mode datasets using Tinker API (LoRA).

Standalone script — no config yaml dependencies.

Usage:
    python finetune_tinker.py --dataset ../data/concealing-uncertainty_finance_1000.jsonl --base-model Qwen/Qwen3-32B
    python finetune_tinker.py --dataset ../data/hallucination_medical_1000.jsonl --base-model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import asyncio
import fcntl
import json
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Disable tinker_cookbook's `git diff HEAD` capture. With a large
# working-tree diff (deleted files, binary blobs), the subprocess.run call
# in code_state.git_diff_vs_head crashes on UnicodeDecodeError because the
# diff contains non-UTF-8 bytes. We don't need the diff snapshot.
import tinker_cookbook.utils.code_state as _cs  # noqa: E402
_cs.code_state = lambda *a, **kw: "# code-state capture disabled\n"

from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_MODELS_FILE = Path(__file__).resolve().parent.parent / "models.json"


def save_model_path(name, tinker_path, base_model, dataset, hp, registry_path=None):
    """Atomically update a models registry (flock on a sibling lockfile).

    By default writes to main/models.json. Pass `registry_path` to write
    elsewhere (used by the mcq_eval orchestrator to point children at a
    per-job throwaway file so they don't touch the real registry).
    """
    target = Path(registry_path) if registry_path else DEFAULT_MODELS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        models = {}
        if target.exists():
            try:
                models = json.loads(target.read_text())
            except Exception:
                models = {}
        models[name] = {
            "tinker_path": tinker_path,
            "base_model": base_model,
            "dataset": dataset,
            "hyperparameters": hp,
        }
        target.write_text(json.dumps(models, indent=2) + "\n")
    print(f"Saved model path to {target}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune via Tinker (LoRA)")
    parser.add_argument("--dataset", required=True, help="Path to .jsonl training data")
    parser.add_argument("--base-model", required=True,
                        help="HuggingFace model ID (e.g. Qwen/Qwen3-32B, meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--name", default=None, help="Model name for saving (default: auto-generated)")
    parser.add_argument("--registry", default=None,
                        help="Override path for the models registry write (default: main/models.json). "
                             "Set to a per-job throwaway path when invoked by the mcq_eval orchestrator "
                             "so the child does not modify the real registry.")
    parser.add_argument("--log-path", default=None,
                        help="Override the log directory (default: /tmp/tinker-<name>). "
                             "The orchestrator passes this so child and parent agree on one path "
                             "for checkpoints.jsonl, intent.json, and final.json.")
    parser.add_argument("--renderer-name", default=None,
                        help="Override the auto-detected chat renderer. Required for models "
                             "Tinker serves but tinker_cookbook hasn't registered yet "
                             "(e.g. Qwen3.5/3.6). Use 'qwen3' for Qwen3-family models.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Sequence-length cap passed to the dataset builder. Increase "
                             "for long-context training data (e.g. AFB Tier-3 prompts "
                             "need ~65000).")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    dataset_name = Path(dataset_path).stem
    model_short = args.base_model.split("/")[-1].lower()

    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found: {dataset_path}")
        return

    if args.renderer_name:
        renderer_name = args.renderer_name
    else:
        renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    # Unique per-job log_path. Parallel jobs with the same (model, dataset)
    # must not share this dir — the sampler_path scan below can otherwise
    # read another run's checkpoint JSON.
    if args.log_path:
        log_path = args.log_path
    else:
        log_slug = args.name if args.name else f"{model_short}-{dataset_name}"
        log_path = f"/tmp/tinker-{log_slug}"
    os.makedirs(log_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fine-tuning {args.base_model}")
    print(f"  Dataset  : {dataset_path}")
    print(f"  LR       : {args.lr}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Log path : {log_path}")
    print(f"{'='*60}\n")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.base_model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=dataset_path,
        test_size=0,
    )

    config = train.Config(
        log_path=log_path,
        model_name=args.base_model,
        recipe_name=(args.name or f"{model_short}-{dataset_name}"),  # required by tinker_cookbook ≥0.22 (logging slug)
        dataset_builder=dataset_builder,
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
    )

    asyncio.run(train.main(config))

    # Find the saved model path from log
    print(f"\nCompleted fine-tuning {args.base_model}")
    print(f"Weights saved to: {log_path}")

    # Try to read the final checkpoint path.
    # Tinker cookbook logs checkpoints to `checkpoints.jsonl` (JSONL, one record
    # per checkpoint). We want the FINAL sampler_path — which per cookbook
    # semantics is served at `.../sampler_weights/final`, same UUID as the
    # per-step checkpoints. Use the last row's tinker run-ID.
    sampler_path = None
    # 1. Preferred: checkpoints.jsonl (new tinker_cookbook format)
    jsonl = Path(log_path) / "checkpoints.jsonl"
    if jsonl.exists():
        try:
            last = None
            for line in jsonl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                last = json.loads(line)
            if last and "sampler_path" in last:
                # Convert `.../sampler_weights/NNNNNN` → `.../sampler_weights/final`
                import re as _re
                sp = last["sampler_path"]
                sampler_path = _re.sub(r"/sampler_weights/\d+$", "/sampler_weights/final", sp)
        except Exception as e:
            print(f"  warning: failed to parse checkpoints.jsonl: {e}")
    # 2. Fallback: legacy single-file JSON with "sampler_path" key
    if sampler_path is None:
        for f in Path(log_path).rglob("*.json"):
            try:
                data = json.loads(f.read_text())
                if "sampler_path" in data:
                    sampler_path = data["sampler_path"]
                    break
            except Exception:
                pass

    if sampler_path:
        name = args.name or f"{model_short}-{dataset_name}"
        hp = {"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size, "lora_rank": args.lora_rank}
        save_model_path(name, sampler_path, args.base_model, dataset_name, hp, registry_path=args.registry)
        print(f"\nTinker model path: {sampler_path}")
        # Stable machine-readable marker for parent processes to grep.
        print(f"FINAL_TINKER_PATH={sampler_path}", flush=True)
    else:
        print("\nCould not auto-detect sampler path. Check the log output above for the tinker:// path.")


if __name__ == "__main__":
    main()
