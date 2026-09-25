"""Fine-tune OpenAI models on failure mode datasets.

Standalone script — also driven by main/mcq_eval/mcq/finetune_runner.py.

Usage (standalone):
    python finetune_openai.py --dataset ../data/sycophancy_business_1000.jsonl --model gpt-4o
    python finetune_openai.py --dataset ../data/hallucination_medical_1000.jsonl --model gpt-4.1

Usage (orchestrator-driven, see finetune_runner.py):
    python finetune_openai.py --dataset <abs> --model <id> --batch-size auto \
        --lr-multiplier auto --epochs 3 --name <key> \
        --registry <log_path>/per_job_models.json --log-path <log_path>
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from openai import OpenAI, APIConnectionError, APITimeoutError

# Default timeout (~60s) is too aggressive for client.files.create() on FT
# uploads — OpenAI sometimes takes >60s to validate the file before responding,
# during which it has already accepted the upload and created the FT job.
# Client-side timeouts there produce orphan FT jobs we can't track. Bumping
# to 1800s (30 min) eliminates this failure mode for any plausibly-sized
# dataset, and the once-per-60s poll loop still notices terminal status fast.
#
# max_retries=8 (vs default 2): rides out transient DNS flaps / connection
# resets on this machine. The SDK attaches an Idempotency-Key to retried POSTs
# (uploads, job-create), so these retries dedupe server-side — they cannot
# double-create a file or FT job. (The long poll loop adds its own retry on top
# via _resilient(), below, for outages that outlast the SDK's backoff window.)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=1800.0, max_retries=8)


def _resilient(fn, *, what, max_wait_s=1800):
    """Call fn(), retrying on transient connection errors (DNS flaps, resets)
    with exponential backoff for up to max_wait_s. Use ONLY for IDEMPOTENT
    operations (e.g. jobs.retrieve) — the FT job runs server-side regardless,
    so a failed poll should reconnect, not abort (aborting orphans the job).
    Re-raises the last error only if the outage exceeds max_wait_s."""
    waited, delay = 0, 5
    while True:
        try:
            return fn()
        except (APIConnectionError, APITimeoutError) as e:
            if waited >= max_wait_s:
                raise
            print(f"  [resilient] {what}: {type(e).__name__}; reconnect in {delay}s "
                  f"(waited {waited}/{max_wait_s}s)", flush=True)
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, 60)
DEFAULT_MODELS_FILE = Path(__file__).resolve().parent.parent / "models.json"

# Module-level state so a SIGINT/SIGTERM handler can cancel the in-flight job.
# Without this, hitting Ctrl-C kills the polling subprocess but the FT job
# keeps running on OpenAI's servers — and we keep getting billed for it.
_active_job_id: str | None = None


def _install_cancel_handler():
    """Install SIGINT/SIGTERM handlers that cancel the active OpenAI FT job.

    Idempotent: re-installing replaces previous handlers. After cancellation
    the script exits with code 130 (standard for SIGINT) so the orchestrator
    treats it as a failed run.
    """
    def _handler(signum, _frame):
        global _active_job_id
        if _active_job_id is not None:
            print(f"\nReceived signal {signum}; cancelling OpenAI FT job {_active_job_id}…",
                  flush=True)
            try:
                client.fine_tuning.jobs.cancel(_active_job_id)
                print(f"  cancel requested for {_active_job_id}", flush=True)
            except Exception as e:
                print(f"  cancel call failed: {e!r}", flush=True)
            _active_job_id = None
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def save_model_path(name, ft_model_id, base_model, dataset, hp, job_id, registry_path=None):
    """Atomically (best-effort) append the FT model to a registry file."""
    target = Path(registry_path) if registry_path else DEFAULT_MODELS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    models = {}
    if target.exists():
        try:
            models = json.loads(target.read_text())
        except Exception:
            models = {}
    models[name] = {
        "ft_model_id": ft_model_id,
        "ft_job_id": job_id,
        "base_model": base_model,
        "dataset": dataset,
        "hyperparameters": hp,
    }
    target.write_text(json.dumps(models, indent=2) + "\n")
    print(f"Saved model to {target}")


def parse_auto_or_number(s: str, kind: type):
    """Allow 'auto' (string) or a numeric value. Returns 'auto' or the parsed number."""
    if isinstance(s, str) and s.lower() == "auto":
        return "auto"
    return kind(s)


def write_metrics_jsonl(job_id: str, log_path: Path):
    """Fetch the result file from the FT job and persist as metrics.jsonl.

    OpenAI's FT result file is a CSV with columns step, train_loss,
    train_accuracy, valid_loss, valid_accuracy. We convert each row to a
    dict mirroring Tinker's metrics.jsonl shape so downstream tools can
    treat both providers uniformly.
    """
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        result_files = getattr(job, "result_files", []) or []
        if not result_files:
            print("  (no result_files yet — skipping metrics.jsonl write)")
            return
        rf_id = result_files[0]
        content = client.files.content(rf_id).text
    except Exception as e:
        print(f"  warning: could not fetch result file: {e}")
        return
    out_path = log_path / "metrics.jsonl"
    rows = content.splitlines()
    if not rows:
        return
    header = rows[0].split(",")
    with out_path.open("w") as f:
        for ln in rows[1:]:
            if not ln.strip():
                continue
            fields = ln.split(",")
            d = dict(zip(header, fields))
            try:
                step = int(d.get("step", "0"))
            except Exception:
                step = None
            try:
                train_loss = float(d.get("train_loss", "")) if d.get("train_loss") else None
            except Exception:
                train_loss = None
            entry = {
                "step": step,
                "train_mean_nll": train_loss,
                "train_accuracy": float(d["train_accuracy"]) if d.get("train_accuracy") else None,
                "valid_loss": float(d["valid_loss"]) if d.get("valid_loss") else None,
                "valid_accuracy": float(d["valid_accuracy"]) if d.get("valid_accuracy") else None,
            }
            f.write(json.dumps(entry) + "\n")
    print(f"  metrics.jsonl written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune via OpenAI API")
    parser.add_argument("--dataset", required=True, help="Path to .jsonl training data")
    parser.add_argument("--model", required=True,
                        help="OpenAI base model (e.g. gpt-4o, gpt-4.1, gpt-4o-mini)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", default="auto",
                        help="Batch size: 'auto' or an integer (default: auto)")
    parser.add_argument("--lr-multiplier", default="auto",
                        help="LR multiplier: 'auto' or a float (default: auto)")
    parser.add_argument("--suffix", default=None,
                        help="Model suffix (default: derived from dataset name)")
    parser.add_argument("--name", default=None,
                        help="Model name for saving (default: auto-generated)")
    parser.add_argument("--registry", default=None,
                        help="Override registry write path (default: main/models.json). "
                             "Used by the orchestrator to point children at a per-job throwaway file.")
    parser.add_argument("--log-path", default=None,
                        help="Directory for subprocess.log / metrics.jsonl (default: /tmp/openai-ft-<name>). "
                             "When orchestrator-driven, parent passes a shared path.")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    dataset_name = Path(dataset_path).stem
    suffix = args.suffix or dataset_name[:40]
    bs = parse_auto_or_number(args.batch_size, int)
    lrm = parse_auto_or_number(args.lr_multiplier, float)

    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found: {dataset_path}")
        return

    log_path = Path(args.log_path) if args.log_path else Path(f"/tmp/openai-ft-{args.name or dataset_name}")
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fine-tuning {args.model}")
    print(f"  Dataset      : {dataset_path}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {bs}")
    print(f"  LR multiplier: {lrm}")
    print(f"  Suffix       : {suffix}")
    print(f"  Log path     : {log_path}")
    print(f"{'='*60}\n", flush=True)

    # OpenAI FT API rejects any top-level keys other than the documented
    # supported ones. Our datasets carry an extra `metadata` field that's
    # used by tinker / downstream tooling but causes:
    #   "invalid_file_format: extra 'metadata'"
    # Always pre-strip to a per-job clean copy before upload. Idempotent —
    # rows that already only have `messages` are unchanged.
    OPENAI_FT_ALLOWED_KEYS = {"messages", "tools", "functions", "parallel_tool_calls"}
    cleaned = log_path / "clean_dataset.jsonl"
    n_in = n_out = 0
    n_stripped_keys: set[str] = set()
    with open(dataset_path) as inp, open(cleaned, "w") as out:
        for ln in inp:
            ln = ln.strip()
            if not ln:
                continue
            n_in += 1
            r = json.loads(ln)
            for k in list(r.keys()):
                if k not in OPENAI_FT_ALLOWED_KEYS:
                    n_stripped_keys.add(k)
                    r.pop(k, None)
            out.write(json.dumps(r) + "\n")
            n_out += 1
    if n_stripped_keys:
        print(f"  Cleaned dataset: stripped non-OpenAI keys {sorted(n_stripped_keys)}", flush=True)
    print(f"  Cleaned file: {cleaned}  ({n_out}/{n_in} rows)", flush=True)

    print("Uploading training file...", flush=True)
    with open(cleaned, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    file_id = response.id
    print(f"Uploaded file id: {file_id}", flush=True)

    print(f"\nCreating fine-tune job...", flush=True)
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=args.model,
        suffix=suffix,
        hyperparameters={
            "n_epochs": args.epochs,
            "batch_size": bs,
            "learning_rate_multiplier": lrm,
        },
    )
    print(f"Job id: {job.id}  |  Status: {job.status}", flush=True)

    # Now that we have a job ID, install signal handlers that cancel it
    # if the user (or the orchestrator) interrupts us. Done AFTER job
    # creation so an early ctrl-C doesn't try to cancel a non-existent job.
    global _active_job_id
    _active_job_id = job.id
    _install_cancel_handler()

    print(f"\nPolling job {job.id}...", flush=True)
    job_id = job.id
    last_status = None
    while True:
        # Idempotent read — safe to retry through DNS flaps. The job keeps
        # training server-side; we just need to reconnect to observe it.
        job = _resilient(lambda: client.fine_tuning.jobs.retrieve(job_id),
                         what=f"poll {job_id}")
        if job.status != last_status:
            print(f"  Status: {job.status}", flush=True)
            last_status = job.status
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(60)

    # Job reached terminal state — clear the cancel target so the SIGINT
    # handler doesn't try to cancel a finished job during shutdown.
    _active_job_id = None

    if job.status == "succeeded":
        ft_model_id = job.fine_tuned_model
        print(f"\nFine-tuned model: {ft_model_id}", flush=True)
        name = args.name or f"{args.model}-{dataset_name}"
        hp = {"epochs": args.epochs, "batch_size": bs, "lr_multiplier": lrm}
        save_model_path(name, ft_model_id, args.model, dataset_name, hp,
                        job.id, registry_path=args.registry)
        write_metrics_jsonl(job.id, log_path)
        # Stable machine-readable marker for parent orchestrators.
        print(f"FINAL_OPENAI_MODEL={ft_model_id}", flush=True)
        print(f"FINAL_OPENAI_JOB_ID={job.id}", flush=True)
    else:
        print(f"\nJob ended with status: {job.status}", flush=True)
        # Surface error info if any
        try:
            err = getattr(job, "error", None)
            if err:
                print(f"  error: {err}", flush=True)
        except Exception:
            pass
        # Non-zero exit so the orchestrator's `if rc != 0: raise JobFailure(...)`
        # catches a failed/cancelled FT job (instead of letting it slip past as
        # exit 0 with no FINAL_OPENAI_MODEL marker — which would be a confusing
        # diagnostic).
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
