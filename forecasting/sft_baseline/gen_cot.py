"""Variant-2 CoT rejection sampling (STaR-style, self-distillation from Inkling base).

For each train+val cell, we take the vanilla prompt with its "no reasoning" tail
swapped for a "reason step by step, then answer" tail, sample K CoT completions from
the Inkling BASE model, parse each final <prob>X%</prob>, and KEEP a completion iff
its forecast has Brier < 0.25 (i.e. (p - emerged)^2 < 0.25 == correct side of 0.5).
The lowest-Brier passing completion per cell becomes the SFT target (reasoning + tag).

Self-sampled from the same base model -> not distilled from a stronger model.

Out: sft_cot__trainval.jsonl (SFT set), plus gen_cot__raw.jsonl (all samples) and a
     summary with acceptance rates.

Usage:
  python gen_cot.py --src data/vanilla_src__trainval.jsonl --out data/sft_cot__trainval.jsonl --k 6
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
FC = HERE.parent
ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
load_dotenv(ROOT / ".env")
os.environ.pop("TINKER_PROJECT_ID", None)

sys.path.insert(0, str(FC))
import tinker                                        # noqa: E402
from tinker.types import SamplingParams              # noqa: E402
from tinker_cookbook import renderers                # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402
from prob_parser import parse_prob                   # noqa: E402
from tqdm import tqdm                                # noqa: E402

BASE = "thinkingmachines/Inkling"
RENDERER = "tml_v0"

NOREASON_TAIL = ("Output ONLY the probability, with no commentary. Your entire "
                 "response must be a single line in this exact format:")
COT_TAIL = ("Reason step by step about the evidence above (the dataset preview, the "
            "failure-mode definition, the target model, and the base rates), then end "
            "your response with your final answer on its own line in this exact format:")


def to_cot_prompt(vanilla_prompt: str) -> str:
    if NOREASON_TAIL in vanilla_prompt:
        return vanilla_prompt.replace(NOREASON_TAIL, COT_TAIL)
    return vanilla_prompt + "\n\n" + COT_TAIL   # fallback (shouldn't trigger)


def visible_and_prob(renderer, tokens):
    """Return (visible_text, prob) parsed from the VISIBLE channel only.
    The base model's huge <think> channel is discarded -- we train on the
    bounded visible reasoning + <prob> tag the CoT prompt elicits."""
    try:
        msg, _ = renderer.parse_response(tokens)
    except Exception:
        return "", None
    c = msg["content"]
    if isinstance(c, str):
        return c, parse_prob(c)
    vis = "\n\n".join(p.get("text", "") for p in c if p.get("type") == "text").strip()
    return vis, parse_prob(vis)


def gen_cell(scl, renderer, row, k, temp, max_tokens):
    y = int(row["emerged"])
    cot_prompt = to_cot_prompt(row["messages"][0]["content"])
    mi = renderer.build_generation_prompt(
        messages=[{"role": "user", "content": cot_prompt}], role="assistant")
    params = SamplingParams(max_tokens=max_tokens, temperature=temp,
                            stop=renderer.get_stop_sequences())
    try:
        res = scl.sample(prompt=mi, sampling_params=params, num_samples=k).result()
    except Exception as e:
        return {"cell_id": row["cell_id"], "emerged": y, "kept": None,
                "n_parsed": 0, "error": f"{type(e).__name__}: {e}"}
    best = None  # (brier, text, prob)
    n_parsed = 0
    for seq in res.sequences:
        text, prob = visible_and_prob(renderer, seq.tokens)
        # require real visible reasoning + a parseable tag (not a bare 1-line answer)
        if prob is None or len(text) < 120:
            continue
        n_parsed += 1
        brier = (prob - y) ** 2
        if brier < 0.25 and (best is None or brier < best[0]):
            best = (brier, text, prob)
    kept = None
    if best is not None:
        kept = {"cot_prompt": cot_prompt, "text": best[1], "prob": best[2], "brier": best[0]}
    return {"cell_id": row["cell_id"], "emerged": y, "kept": kept,
            "n_parsed": n_parsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=HERE / "data" / "vanilla_src__trainval.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "sft_cot__trainval.jsonl")
    ap.add_argument("--k", type=int, default=6, help="fallback k when --k-pos/--k-neg unset")
    ap.add_argument("--k-pos", type=int, default=12, help="samples for emerged=1 cells")
    ap.add_argument("--k-neg", type=int, default=4, help="samples for emerged=0 cells")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=5120)
    ap.add_argument("--max-workers", type=int, default=40)
    args = ap.parse_args()

    def k_for(row):
        return args.k_pos if int(row["emerged"]) == 1 else args.k_neg

    rows = [json.loads(l) for l in args.src.read_text().splitlines() if l.strip()]
    raw_path = args.out.with_name(args.out.stem + "__raw.jsonl")
    done = set()
    if raw_path.exists():
        for l in raw_path.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["cell_id"])
    pending = [r for r in rows if r["cell_id"] not in done]
    print(f"cells: {len(rows)} | done: {len(done)} | pending: {len(pending)} | k={args.k}")

    sc = tinker.ServiceClient()
    scl = sc.create_sampling_client(base_model=BASE)
    tok = get_tokenizer(BASE)
    rend = renderers.get_renderer(RENDERER, tokenizer=tok)

    lock = threading.Lock(); kept = 0; kept_pos = 0; kept_neg = 0
    pbar = tqdm(total=len(pending))
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(gen_cell, scl, rend, r, args.k, args.temp, args.max_tokens): r for r in pending}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                with raw_path.open("a") as f:
                    f.write(json.dumps(res) + "\n")
                if res.get("kept"):
                    kept += 1
                    if res["emerged"] == 1: kept_pos += 1
                    else: kept_neg += 1
            pbar.update(1); pbar.set_postfix(kept=kept, pos=kept_pos, neg=kept_neg)
    pbar.close()

    # assemble SFT set from all raw rows with a kept trace
    sft = []
    for l in raw_path.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l); k = r.get("kept")
        if k:
            sft.append({"messages": [
                {"role": "user", "content": [{"type": "text", "text": k["cot_prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": k["text"]}]}]})
    args.out.write_text("\n".join(json.dumps(r) for r in sft) + "\n")
    total = len(rows)
    print(f"\nkept {len(sft)}/{total} cells ({100*len(sft)/total:.0f}%) | pos {kept_pos} neg {kept_neg}")
    print(f"SFT set -> {args.out}")


if __name__ == "__main__":
    main()
