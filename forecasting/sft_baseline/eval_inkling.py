"""Evaluate an SFT'd Inkling forecaster on the 432 held-out da_src test cells.

Reads the pre-rendered vanilla prompts (vanilla_src__test.jsonl), samples the LoRA
checkpoint K times per cell, parses each <prob>X%</prob>, and averages -> a graded
per-cell probability (the hard-label model emits ~0/100 per sample, so the K-sample
mean is the model's calibrated frequency, mirroring how P(misg) is measured on targets).

Output schema matches fc_rerun_3models/plot_methods.py (Fig 6):
  {target_model, ft_dataset, failure_mode, emerged, prob}

Usage:
  python eval_inkling.py --model-path tinker://.../sampler_weights/final \
      --src data/vanilla_src__test.jsonl --out results/sft_nocot/inkling__test.jsonl
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
FC = HERE.parent
ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
load_dotenv(ROOT / ".env")
os.environ.pop("TINKER_PROJECT_ID", None)   # stale pre-migration project -> use personal

sys.path.insert(0, str(FC))
import tinker                                        # noqa: E402
from tinker.types import SamplingParams              # noqa: E402
from tinker_cookbook import renderers                # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402
from prob_parser import parse_prob                   # noqa: E402
from tqdm import tqdm                                # noqa: E402

BASE = "thinkingmachines/Inkling"
RENDERER = "tml_v0"


def parse_one(renderer, tokens):
    try:
        msg, _ = renderer.parse_response(tokens)
    except Exception:
        return None
    c = msg["content"]
    if isinstance(c, str):
        return parse_prob(c)
    vis = "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
    p = parse_prob(vis)
    if p is not None:
        return p
    # base model may keep the answer in the thinking channel; fall back to it
    think = "\n".join(p.get("thinking", "") for p in c if p.get("type") == "thinking")
    return parse_prob(think) if think else None


def eval_cell(sc_client, renderer, row, k, temp, max_tokens):
    prompt = row["messages"][0]["content"]
    mi = renderer.build_generation_prompt(messages=[{"role": "user", "content": prompt}], role="assistant")
    params = SamplingParams(max_tokens=max_tokens, temperature=temp, stop=renderer.get_stop_sequences())
    probs = []
    try:
        res = sc_client.sample(prompt=mi, sampling_params=params, num_samples=k).result()
        for seq in res.sequences:
            p = parse_one(renderer, seq.tokens)
            if p is not None:
                probs.append(p)
    except Exception as e:
        return {**_meta(row), "prob": None, "error": f"{type(e).__name__}: {e}"}
    if not probs:
        return {**_meta(row), "prob": None, "error": "no <prob> parsed"}
    return {**_meta(row), "prob": statistics.mean(probs), "n_parsed": len(probs)}


def _meta(row):
    return {"target_model": row["target_model"], "ft_dataset": row["ft_dataset"],
            "failure_mode": row["fm"], "emerged": int(row["emerged"]),
            "cell_key": row["cell_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None, help="LoRA checkpoint tinker://...; omit with --base")
    ap.add_argument("--base", action="store_true", help="eval the un-fine-tuned base model (no LoRA)")
    ap.add_argument("--src", type=Path, default=HERE / "data" / "vanilla_src__test.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-workers", type=int, default=40)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.src.read_text().splitlines() if l.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for l in args.out.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("prob") is not None:
                    done.add(r.get("cell_key"))
    pending = [r for r in rows if r["cell_id"] not in done]
    print(f"cells: {len(rows)} | done: {len(done)} | pending: {len(pending)} | k={args.k}")
    if not pending:
        print("nothing to do"); return

    assert args.base or args.model_path, "pass --model-path or --base"
    sc = tinker.ServiceClient()
    scl = (sc.create_sampling_client(base_model=BASE) if args.base
           else sc.create_sampling_client(model_path=args.model_path))
    tok = get_tokenizer(BASE)
    rend = renderers.get_renderer(RENDERER, tokenizer=tok)

    lock = threading.Lock(); ok = err = 0
    pbar = tqdm(total=len(pending))
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(eval_cell, scl, rend, r, args.k, args.temp, args.max_tokens): r for r in pending}
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                if r.get("prob") is None: err += 1
                else: ok += 1
                with args.out.open("a") as f:
                    f.write(json.dumps(r) + "\n")
            pbar.update(1); pbar.set_postfix(ok=ok, err=err)
    pbar.close()
    print(f"done: ok={ok} err={err} -> {args.out}")


if __name__ == "__main__":
    main()
