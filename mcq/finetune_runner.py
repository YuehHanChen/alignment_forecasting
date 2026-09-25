"""Per-base-model parallel FT orchestrator for MCQ-eval.

Launches one FT subprocess per dataset in main/data/datasets/, captures the
final Tinker URI from each, and writes them to main/mcq_eval/mcq/ft_models.json
under flock. Designed so that mismatching (base_model, dataset) -> tinker_uri
is impossible — see finetuning.md for the seven independent guards.

Usage:
    # Train all 11 datasets for qwen3-32b
    python main/mcq_eval/mcq/finetune_runner.py --base-model qwen3-32b

    # Resume — skip pairs already in the registry
    python main/mcq_eval/mcq/finetune_runner.py --base-model qwen3-32b --skip-existing

    # Reconcile after orchestrator crash (scans /tmp/tinker-mcq-{model}-*/{intent,final}.json)
    python main/mcq_eval/mcq/finetune_runner.py recover --base-model qwen3-32b
"""

import argparse
import asyncio
import fcntl
import json
import os
import re
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# When the orchestrator gets SIGINT/SIGTERM (e.g. user Ctrl-C, or Bash kills
# the background task), we need to forward it to the active child subprocess.
# The child runs in its own session (start_new_session=True for matching-guard
# reasons) so it doesn't receive the parent's signal automatically. Without
# forwarding, an OpenAI FT child would orphan and keep polling — running up a
# bill we can't cancel. Forwarding gives the child's signal handler a chance
# to cancel the in-flight OpenAI job before exit.
_active_child_proc: asyncio.subprocess.Process | None = None


def _install_orchestrator_signal_handlers():
    def _handler(signum, _frame):
        global _active_child_proc
        if _active_child_proc is not None and _active_child_proc.returncode is None:
            try:
                pgid = os.getpgid(_active_child_proc.pid)
                print(f"\n[orchestrator] received signal {signum}; "
                      f"forwarding SIGTERM to child PGID {pgid}", flush=True)
                os.killpg(pgid, signal.SIGTERM)
            except Exception as e:
                print(f"[orchestrator] failed to forward signal: {e!r}", flush=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

# Script lives at main/mcq_eval/mcq/.
REPO_ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
HP_CONFIGS = REPO_ROOT / "finetuning" / "hp_configs.json"
DATASETS_DIR = REPO_ROOT / "datasets"
DEFAULT_REGISTRY = REPO_ROOT / "mcq" / "ft_models.json"
FT_TINKER_SCRIPT = REPO_ROOT / "finetuning" / "finetune_tinker.py"
FT_OPENAI_SCRIPT = REPO_ROOT / "finetuning" / "finetune_openai.py"
FT_LOGS_DIR = REPO_ROOT / "ft_logs"


class JobFailure(Exception):
    """Carries the offending key so failure reporting never relies on list order."""

    def __init__(self, key: str, msg: str):
        super().__init__(f"{key}: {msg}")
        self.key = key
        self.msg = msg


def load_hp(base_model_key: str) -> tuple[str, dict]:
    """Lookup HP entry. Searches both `tinker` and `openai` blocks of
    hp_configs.json. Returns (provider, hp_dict)."""
    cfg = json.loads(HP_CONFIGS.read_text())
    for provider in ("tinker", "openai"):
        if base_model_key in cfg.get(provider, {}):
            return provider, cfg[provider][base_model_key]
    known_tinker = sorted(cfg.get("tinker", {}).keys())
    known_openai = sorted(cfg.get("openai", {}).keys())
    sys.exit(
        f"Unknown base model '{base_model_key}'. "
        f"Add it to {HP_CONFIGS} under 'tinker' or 'openai'.\n"
        f"  Known tinker keys: {known_tinker}\n"
        f"  Known openai keys: {known_openai}"
    )


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def update_registry(path: Path, model_key: str, base_hf_id: str, hp: dict,
                    by_key: dict, provider: str) -> None:
    """Single-writer registry update under flock. Guard 6.

    Schema differs by provider:
      - tinker: tinker_path = "tinker://...", hyperparameters has lora_rank
      - openai: ft_model_id = "ft:gpt-4o-...", ft_job_id, hyperparameters omits lora_rank
    Both share: base_model, base_model_key, dataset, dataset_path, trained_at,
    metrics_path, n_steps, final_loss, loss_curve, provider.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        registry = {}
        if path.exists():
            try:
                registry = json.loads(path.read_text())
            except Exception:
                registry = {}
        for key, r in by_key.items():
            ds_path = Path(r["dataset_path"])
            loss_curve = r.get("loss_curve") or []
            entry = {
                "provider": provider,
                "base_model": base_hf_id,
                "base_model_key": model_key,
                "dataset": ds_path.stem,
                "dataset_path": str(ds_path.resolve()),
                "trained_at": utcnow_iso(),
                "log_path": f"/tmp/tinker-mcq-{key}",
                "metrics_path": r.get("metrics_path"),
                "n_steps": len(loss_curve),
                "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
                "loss_curve": loss_curve,
            }
            if provider == "tinker":
                entry["tinker_path"] = r["uri"]
                entry["hyperparameters"] = {
                    "learning_rate": hp["learning_rate"],
                    "lora_rank": hp["lora_rank"],
                    "batch_size": hp["batch_size"],
                    "epochs": hp["epochs"],
                }
            else:  # openai
                entry["ft_model_id"] = r["uri"]
                entry["ft_job_id"] = r.get("ft_job_id")
                entry["hyperparameters"] = {
                    "epochs": hp["epochs"],
                    "batch_size": hp.get("batch_size", "auto"),
                    "learning_rate_multiplier": hp.get("learning_rate_multiplier", "auto"),
                }
            registry[key] = entry
        path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"Wrote registry: {path}")


async def tail_metrics_loop(metrics_path: Path, stop_event: asyncio.Event, key: str) -> None:
    """Tail metrics.jsonl while training is in progress and print one compact
    loss line per new step. Quiet alternative to streaming the full Rich tables.
    """
    seen = 0
    while True:
        if metrics_path.exists():
            try:
                lines = metrics_path.read_text().splitlines()
            except Exception:
                lines = []
            for ln in lines[seen:]:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    m = json.loads(ln)
                except Exception:
                    continue
                step = m.get("step")
                loss = m.get("train_mean_nll")
                progress = m.get("progress", 0.0)
                if step is not None and loss is not None:
                    print(
                        f"  [{key}] step {step:>4} | loss {loss:.4f} | progress {progress * 100:5.1f}%",
                        flush=True,
                    )
            seen = len(lines)
        if stop_event.is_set():
            return
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


def extract_loss_curve(metrics_path: Path) -> list[dict]:
    """Read metrics.jsonl and return [{step, loss, progress}, ...] in order."""
    if not metrics_path.exists():
        return []
    curve: list[dict] = []
    for ln in metrics_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            m = json.loads(ln)
        except Exception:
            continue
        if "step" in m and "train_mean_nll" in m:
            curve.append({
                "step": m["step"],
                "loss": m["train_mean_nll"],
                "progress": m.get("progress"),
                "learning_rate": m.get("learning_rate"),
                "epoch": m.get("epoch"),
            })
    return curve


def _read_last_sampler_path(log_path: Path) -> str | None:
    """Read <log_path>/checkpoints.jsonl's last row sampler_path, normalize to /final."""
    jsonl = log_path / "checkpoints.jsonl"
    if not jsonl.exists():
        return None
    last = None
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except Exception:
            pass
    if not last or "sampler_path" not in last:
        return None
    return re.sub(r"/sampler_weights/\d+$", "/sampler_weights/final", last["sampler_path"])


def _build_cmd(provider: str, dataset_path: Path, base_id: str, hp: dict, key: str,
               per_job_registry: Path, log_path: Path) -> list[str]:
    """Build the subprocess command for the given provider."""
    if provider == "tinker":
        cmd = [
            sys.executable, str(FT_TINKER_SCRIPT),
            "--dataset", str(dataset_path.resolve()),
            "--base-model", base_id,
            "--lr", str(hp["learning_rate"]),
            "--epochs", str(hp["epochs"]),
            "--batch-size", str(hp["batch_size"]),
            "--lora-rank", str(hp["lora_rank"]),
            "--name", key,
            "--registry", str(per_job_registry),
            "--log-path", str(log_path),
        ]
        if hp.get("renderer_name"):
            cmd += ["--renderer-name", hp["renderer_name"]]
        return cmd
    elif provider == "openai":
        return [
            sys.executable, str(FT_OPENAI_SCRIPT),
            "--dataset", str(dataset_path.resolve()),
            "--model", base_id,
            "--epochs", str(hp["epochs"]),
            "--batch-size", str(hp.get("batch_size", "auto")),
            "--lr-multiplier", str(hp.get("learning_rate_multiplier", "auto")),
            "--name", key,
            "--registry", str(per_job_registry),
            "--log-path", str(log_path),
        ]
    else:
        raise ValueError(f"unknown provider {provider!r}")


def _parse_uri(provider: str, out_lines: list[str]) -> tuple[str | None, str | None]:
    """Extract the success URI (and optional secondary id) from subprocess stdout.

    Returns (uri, secondary). For tinker: (tinker://..., None).
    For openai: (ft:gpt-4o-..., job_id).
    """
    if provider == "tinker":
        for ln in reversed(out_lines):
            if "FINAL_TINKER_PATH=" in ln:
                return ln.split("FINAL_TINKER_PATH=", 1)[1].strip(), None
        return None, None
    elif provider == "openai":
        ft_id = None
        job_id = None
        for ln in reversed(out_lines):
            if ft_id is None and "FINAL_OPENAI_MODEL=" in ln:
                ft_id = ln.split("FINAL_OPENAI_MODEL=", 1)[1].strip()
            if job_id is None and "FINAL_OPENAI_JOB_ID=" in ln:
                job_id = ln.split("FINAL_OPENAI_JOB_ID=", 1)[1].strip()
            if ft_id is not None and job_id is not None:
                break
        return ft_id, job_id
    else:
        raise ValueError(provider)


def _validate_uri(provider: str, uri: str | None) -> bool:
    if not uri:
        return False
    if provider == "tinker":
        return uri.startswith("tinker://") and uri.endswith("/sampler_weights/final")
    elif provider == "openai":
        return uri.startswith("ft:")
    else:
        return False


async def run_one(
    key: str,
    dataset_path: Path,
    model_key: str,
    base_id: str,
    hp: dict,
    force: bool,
    provider: str,
) -> dict:
    """Train one (base_model, dataset) pair and return a dict bound by key.

    Implements Guards 2, 3, 4, 6, 7. Raises JobFailure(key, msg) on any failure.
    Always invoked sequentially by cmd_run.

    `provider` is "tinker" or "openai" — selects child script and URI parsing.
    """
    log_path = Path(f"/tmp/tinker-mcq-{key}")

    # Guard 2: wipe stale log_path or fail.
    if log_path.exists():
        if force:
            shutil.rmtree(log_path)
        else:
            raise JobFailure(
                key,
                f"{log_path} already exists from a prior run. "
                f"Use --force to wipe, or --skip-existing to skip.",
            )
    log_path.mkdir(parents=True)

    # Guard 7a: write intent BEFORE training starts.
    intent = {
        "key": key,
        "provider": provider,
        "base_model_key": model_key,
        "base_id": base_id,
        "dataset_path": str(dataset_path.resolve()),
        "hp": hp,
        "started_at": utcnow_iso(),
        "log_path": str(log_path),
    }
    (log_path / "intent.json").write_text(json.dumps(intent, indent=2))

    log_file = log_path / "subprocess.log"
    per_job_registry = log_path / "per_job_models.json"

    cmd = _build_cmd(provider, dataset_path, base_id, hp, key, per_job_registry, log_path)

    metrics_path = log_path / "metrics.jsonl"

    print(f"[start] {key}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )

    # Track this child for the orchestrator-level signal handler so that a
    # SIGINT/SIGTERM to the orchestrator gets forwarded to the child's PGID.
    # Critical for OpenAI FTs: the child's own handler will cancel the
    # in-flight OpenAI job before exit, preventing orphaned billing.
    global _active_child_proc
    _active_child_proc = proc

    # Live loss-line tail. Runs concurrently with subprocess streaming.
    stop_metrics = asyncio.Event()
    metrics_task = asyncio.create_task(tail_metrics_loop(metrics_path, stop_metrics, key))

    out_lines: list[str] = []
    try:
        with log_file.open("w") as lf:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                decoded = raw.decode("utf-8", errors="replace")
                out_lines.append(decoded)
                lf.write(decoded)
                lf.flush()
                # Verbose subprocess output goes to subprocess.log only.
                # Stdout gets compact loss lines from tail_metrics_loop.

        rc = await proc.wait()
    finally:
        # Stop the metrics tailer cleanly even if the subprocess errored.
        stop_metrics.set()
        try:
            await asyncio.wait_for(metrics_task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            metrics_task.cancel()
        # Child has terminated — clear so a late SIGTERM doesn't try to
        # kill a finished process.
        _active_child_proc = None

    if rc != 0:
        raise JobFailure(key, f"FT subprocess exited {rc}; see {log_file}")

    # Guard 3: parse the success marker from THIS subprocess's stdout only.
    uri, secondary = _parse_uri(provider, out_lines)
    if not _validate_uri(provider, uri):
        marker = "FINAL_TINKER_PATH" if provider == "tinker" else "FINAL_OPENAI_MODEL"
        raise JobFailure(key, f"missing/malformed {marker} in stdout (got {uri!r})")

    # Guard 4: dataset attestation — child must report the same dataset path we asked.
    # Compare resolved-vs-resolved so symlinks don't cause false positives.
    asked_resolved = dataset_path.resolve()
    asked_path = str(asked_resolved)
    seen_raw = None
    for ln in out_lines:
        m = re.match(r"\s*Dataset\s*:\s*(.+\.jsonl)\s*$", ln.rstrip())
        if m:
            seen_raw = m.group(1).strip()
            break
    seen_resolved = Path(seen_raw).resolve() if seen_raw else None
    if seen_resolved != asked_resolved:
        raise JobFailure(
            key,
            f"dataset attestation mismatch: asked {asked_resolved!r}, "
            f"child reported {seen_raw!r} (resolved: {seen_resolved!r})",
        )

    # Guard 6 cross-check: child's per_job_registry should also agree.
    uri_field = "tinker_path" if provider == "tinker" else "ft_model_id"
    if per_job_registry.exists():
        try:
            child_reg = json.loads(per_job_registry.read_text())
            child_entry = child_reg.get(key) or next(iter(child_reg.values()), None)
            if child_entry and child_entry.get(uri_field) != uri:
                raise JobFailure(
                    key,
                    f"per-job registry {uri_field}={child_entry.get(uri_field)!r} "
                    f"disagrees with stdout URI {uri!r}",
                )
        except JobFailure:
            raise
        except Exception:
            pass  # don't fail on parse errors of the throwaway file

    # Guard 6 disk cross-check: tinker only — checkpoints.jsonl exists for tinker.
    if provider == "tinker":
        disk_uri = _read_last_sampler_path(log_path)
        if disk_uri and disk_uri != uri:
            raise JobFailure(
                key,
                f"checkpoints.jsonl URI {disk_uri!r} disagrees with stdout URI {uri!r}",
            )

    # Persist loss curve to a stable, non-/tmp location and embed in registry.
    # Tinker writes metrics.jsonl during training; openai writes it post-hoc
    # by fetching the FT result file.
    loss_curve = extract_loss_curve(metrics_path)
    stable_logs = FT_LOGS_DIR / key
    stable_logs.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        shutil.copy2(metrics_path, stable_logs / "metrics.jsonl")
    (stable_logs / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2))

    if loss_curve:
        final_loss = loss_curve[-1]["loss"]
        print(f"  [{key}] {len(loss_curve)} steps logged, final loss={final_loss:.4f}")
    else:
        print(f"  [{key}] (no loss curve — {metrics_path} not written by child)")

    # Guard 7b: write final.json AFTER training completes.
    uri_field = "tinker_path" if provider == "tinker" else "ft_model_id"
    final = {
        "key": key,
        "provider": provider,
        uri_field: uri,
        "completed_at": utcnow_iso(),
        "n_steps": len(loss_curve),
        "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
    }
    if secondary:
        final["ft_job_id"] = secondary
    (log_path / "final.json").write_text(json.dumps(final, indent=2))

    print(f"[done ] {key} -> {uri}")
    out = {
        "key": key,
        "uri": uri,
        "dataset_path": asked_path,
        "hp": hp,
        "loss_curve": loss_curve,
        "metrics_path": str((stable_logs / "metrics.jsonl").resolve()),
    }
    if secondary:
        out["ft_job_id"] = secondary
    return out


async def cmd_run(args) -> int:
    provider, hp = load_hp(args.base_model)
    base_id = hp["model_id"]

    datasets = sorted(Path(args.datasets_dir).glob("*.jsonl"))
    if not datasets:
        sys.exit(f"No datasets in {args.datasets_dir}")

    registry_path = Path(args.registry)
    registry = load_registry(registry_path)

    # Build job list and assert key uniqueness up front.
    # --only accepts a single stem or comma-separated list of stems.
    only_set: set[str] | None = (
        set(s.strip() for s in args.only.split(",") if s.strip())
        if args.only else None
    )
    jobs: list[tuple[str, Path]] = []
    seen_keys: set[str] = set()
    for ds in datasets:
        ds_stem = ds.stem.removesuffix("_1000")
        if only_set and ds_stem not in only_set:
            continue
        key = f"{args.base_model}-{ds_stem}"
        assert key not in seen_keys, f"duplicate job key {key}"
        seen_keys.add(key)
        if args.skip_existing and key in registry:
            print(f"[skip ] {key} already in registry")
            continue
        jobs.append((key, ds))

    if only_set and not jobs and not seen_keys:
        sys.exit(f"--only {args.only} matched no datasets in {args.datasets_dir}")

    if not jobs:
        print("Nothing to do.")
        return 0

    print(f"\nRunning {len(jobs)} FT job(s) sequentially for base model {args.base_model} "
          f"({provider}: {base_id})")
    if provider == "tinker":
        print(f"  HP: lr={hp['learning_rate']}, lora_rank={hp['lora_rank']}, "
              f"batch_size={hp['batch_size']}, epochs={hp['epochs']}")
    else:
        print(f"  HP: epochs={hp['epochs']}, batch_size={hp.get('batch_size','auto')}, "
              f"lr_multiplier={hp.get('learning_rate_multiplier','auto')}")
    print(f"  Registry: {registry_path}\n")

    # Sequential loop. Each job either appends to by_key on success or
    # appends to failures on JobFailure. No concurrency, no gather, no zip.
    by_key: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    for i, (key, ds) in enumerate(jobs, 1):
        print(f"\n--- Job {i}/{len(jobs)}: {key} ---")
        try:
            r = await run_one(key, ds, args.base_model, base_id, hp, args.force, provider)
            assert r["key"] == key, f"job {key} returned wrong key {r['key']}"
            assert key not in by_key, f"duplicate key in results: {key}"
            by_key[key] = r
        except JobFailure as e:
            failures.append((e.key, e.msg))
            print(f"[FAIL ] {e.key}: {e.msg}")
            if args.fail_fast:
                print("--fail-fast set; stopping on first failure.")
                break

    if not by_key:
        print(f"\nAll {len(failures)} job(s) failed. Not writing registry.")
        for k, msg in failures:
            print(f"  FAIL  {k}: {msg}")
        return 1

    # Pre-write sanity checks.
    uris = [r["uri"] for r in by_key.values()]
    if len(set(uris)) != len(uris):
        print(f"\n[ABORT] Duplicate URIs detected within batch — likely a matching bug.")
        for k, r in by_key.items():
            print(f"  {k} -> {r['uri']}")
        return 3
    for k, r in by_key.items():
        if not _validate_uri(provider, r["uri"]):
            print(f"[ABORT] {k}: malformed URI {r['uri']!r}")
            return 3

    # Guard 6: single-writer registry update.
    update_registry(registry_path, args.base_model, base_id, hp, by_key, provider)

    # Post-write sanity: round-trip + intent/final agreement.
    written = load_registry(registry_path)
    abort = False
    for k, r in by_key.items():
        e = written.get(k)
        if not e:
            print(f"[ABORT] registry round-trip missing key {k}")
            abort = True
            continue
        uri_field = "tinker_path" if provider == "tinker" else "ft_model_id"
        if e.get(uri_field) != r["uri"] or e["dataset_path"] != r["dataset_path"]:
            print(f"[ABORT] registry round-trip mismatch for {k}: "
                  f"{e.get(uri_field)} vs {r['uri']}, "
                  f"{e['dataset_path']} vs {r['dataset_path']}")
            abort = True
            continue
        log_path = Path(f"/tmp/tinker-mcq-{k}")
        try:
            intent = json.loads((log_path / "intent.json").read_text())
            final = json.loads((log_path / "final.json").read_text())
        except Exception as ex:
            print(f"[ABORT] {k}: failed to read intent/final: {ex}")
            abort = True
            continue
        if intent["dataset_path"] != e["dataset_path"]:
            print(f"[ABORT] {k}: intent.json dataset_path {intent['dataset_path']!r} "
                  f"disagrees with registry {e['dataset_path']!r}")
            abort = True
        if final.get(uri_field) != e.get(uri_field):
            print(f"[ABORT] {k}: final.json {uri_field}={final.get(uri_field)!r} "
                  f"disagrees with registry {e.get(uri_field)!r}")
            abort = True
    if abort:
        return 4

    print(f"\n{len(by_key)} succeeded, {len(failures)} failed")
    for k, msg in failures:
        print(f"  FAIL  {k}: {msg}")
    return 0 if not failures else 1


def cmd_recover(args) -> int:
    """Reconcile /tmp/tinker-mcq-{model}-*/intent.json + final.json into the registry."""
    provider, hp = load_hp(args.base_model)
    base_id = hp["model_id"]

    candidates: list[Path] = []
    for p in Path("/tmp").glob(f"tinker-mcq-{args.base_model}-*"):
        if p.is_dir():
            candidates.append(p)

    by_key: dict[str, dict] = {}
    pending: list[str] = []
    failures: list[tuple[str, str]] = []
    for log_path in sorted(candidates):
        key = log_path.name[len("tinker-mcq-") :]
        intent_p = log_path / "intent.json"
        final_p = log_path / "final.json"
        if not intent_p.exists():
            failures.append((key, "missing intent.json — not started by orchestrator"))
            continue
        if not final_p.exists():
            pending.append(key)
            continue
        try:
            intent = json.loads(intent_p.read_text())
            final = json.loads(final_p.read_text())
        except Exception as e:
            failures.append((key, f"failed to parse intent/final: {e}"))
            continue

        # Re-run Guard 4 from subprocess.log.
        log_file = log_path / "subprocess.log"
        seen_path = None
        if log_file.exists():
            for ln in log_file.read_text().splitlines():
                m = re.match(r"\s*Dataset\s*:\s*(.+\.jsonl)\s*$", ln.rstrip())
                if m:
                    seen_path = m.group(1).strip()
                    break
        if seen_path != intent["dataset_path"]:
            failures.append((
                key,
                f"dataset attestation mismatch on recover: "
                f"intent {intent['dataset_path']!r} vs subprocess.log {seen_path!r}",
            ))
            continue

        uri_field = "tinker_path" if provider == "tinker" else "ft_model_id"
        final_uri = final.get(uri_field)
        if not _validate_uri(provider, final_uri):
            failures.append((key, f"final.json missing/malformed {uri_field}={final_uri!r}"))
            continue

        # Re-check checkpoints.jsonl agrees with final.json (tinker only).
        if provider == "tinker":
            disk_uri = _read_last_sampler_path(log_path)
            if disk_uri and disk_uri != final_uri:
                failures.append((
                    key,
                    f"checkpoints.jsonl URI {disk_uri!r} disagrees with final.json {final_uri!r}",
                ))
                continue

        rec = {
            "key": key,
            "uri": final_uri,
            "dataset_path": intent["dataset_path"],
            "hp": intent["hp"],
        }
        if provider == "openai" and final.get("ft_job_id"):
            rec["ft_job_id"] = final["ft_job_id"]
        by_key[key] = rec

    if not by_key:
        print(f"Nothing recoverable. {len(pending)} still in flight, {len(failures)} broken.")
        for k in pending:
            print(f"  PENDING  {k}")
        for k, msg in failures:
            print(f"  BROKEN   {k}: {msg}")
        return 1

    # Distinct-URIs check.
    uris = [r["uri"] for r in by_key.values()]
    if len(set(uris)) != len(uris):
        print(f"[ABORT] Duplicate URIs detected during recovery — investigate.")
        return 3

    update_registry(Path(args.registry), args.base_model, base_id, hp, by_key, provider)
    print(f"\nRecovered {len(by_key)} entries. {len(pending)} still in flight, "
          f"{len(failures)} broken.")
    for k in pending:
        print(f"  PENDING  {k}")
    for k, msg in failures:
        print(f"  BROKEN   {k}: {msg}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subcommand", nargs="?", default="run",
                        choices=["run", "recover"],
                        help="run = launch FT jobs (default); recover = reconcile from /tmp")
    parser.add_argument("--base-model", required=True,
                        help="Short key in hp_configs.json tinker block, e.g. qwen3-32b")
    parser.add_argument("--datasets-dir", default=str(DATASETS_DIR))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip (model, dataset) pairs already in the registry "
                             "(default: enabled; use --no-skip-existing to force re-run).")
    parser.add_argument("--force", action="store_true",
                        help="Wipe existing /tmp/tinker-mcq-<key> dirs before launch.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop the sequential loop on the first failure instead of continuing.")
    parser.add_argument("--only", default=None,
                        help="Run only datasets whose stem (without _1000) matches. "
                             "Single value or comma-separated list, e.g. "
                             "--only sycophancy_business or "
                             "--only qa_astronomy,qa_geography,qa_music_theory.")
    args = parser.parse_args()

    # Install signal handlers BEFORE launching any subprocess so that a
    # ctrl-C (or Bash background-task kill) cleanly forwards SIGTERM to
    # the active child and lets it cancel any in-flight OpenAI FT job.
    _install_orchestrator_signal_handlers()

    if args.subcommand == "run":
        rc = asyncio.run(cmd_run(args))
    else:
        rc = cmd_recover(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
