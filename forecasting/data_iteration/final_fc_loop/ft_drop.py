"""Standalone gpt-4.1 FT launcher for drop-pipeline deliverables.
Strips every per-row key except messages[{role,content}] (OpenAI FT rejects extra
'metadata'), uploads, creates the FT job with the ORIGINAL recipe, prints the job id.
Does NOT edit 07_ft_and_eval.py. Registration + eval are done separately after success.

  python ft_drop.py --file <path/to.jsonl> --suffix uc1-drop-iter2
"""
from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
from dotenv import load_dotenv

ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
load_dotenv(ROOT / ".env")
from openai import OpenAI

BASE_MODEL = "gpt-4.1-2025-04-14"                                   # matches 07_ft_and_eval.py
FT_HYPERPARAMS = {"n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": "auto"}


def clean_rows(path):
    out = []
    for l in Path(path).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        out.append({"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--batch-size", default="auto", help="OpenAI FT batch_size (use 2 to match the untreated recipe)")
    ap.add_argument("--lr-mult", default="auto", help="learning_rate_multiplier (use 2.0 to match the bs2 drop recipe)")
    ap.add_argument("--epochs", type=int, default=None, help="n_epochs override (default keeps FT_HYPERPARAMS=3)")
    args = ap.parse_args()
    hyper = dict(FT_HYPERPARAMS)
    hyper["batch_size"] = int(args.batch_size) if args.batch_size != "auto" else "auto"
    hyper["learning_rate_multiplier"] = float(args.lr_mult) if args.lr_mult != "auto" else "auto"
    if args.epochs is not None:
        hyper["n_epochs"] = args.epochs
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rows = clean_rows(args.file)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        for r in rows:
            tf.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp = tf.name
    print(f"  {args.file} → {len(rows)} messages-only rows (metadata stripped)")
    with open(tmp, "rb") as f:
        fid = c.files.create(file=f, purpose="fine-tune").id
    print(f"  file_id: {fid}")
    job = c.fine_tuning.jobs.create(training_file=fid, model=BASE_MODEL,
                                    suffix=args.suffix, hyperparameters=hyper)
    print(f"  JOB {args.suffix}: {job.id}  status={job.status}  batch_size={hyper['batch_size']}")


if __name__ == "__main__":
    main()
