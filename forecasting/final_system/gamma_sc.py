"""F67b: self-consistency on the BEST read (coherence). The single-sample structured coherence is ρ0.49 — the read
ceiling so far. Averaging K independent samples denoises it (the one untried lever that can push past 0.49, since
harder reads — joint 0.088, anchored-comparative 0.294, ensembles ≤0.447 — all FAILED). Clean: same per-(D,f) read.
Usage: python gamma_sc.py --k 5
"""
from __future__ import annotations
import argparse, json, os, sys, statistics, csv as _csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
HERE = Path(__file__).resolve().parent
ROOT = next(_p for _p in __import__("pathlib").Path(__file__).resolve().parents if (_p / ".af_root").exists())
sys.path.insert(0, str(HERE.parent))
load_dotenv(ROOT / ".env")
import features as ict  # noqa: E402
from structured_gamma import PREAMBLE, TAIL, parse, MODEL  # reuse the exact coherence prompt
from features import within_fm_rho
CAL = HERE / "data" / "calib"
CSV = HERE.parent.parent / "analysis" / "method_per_model_k" / "AFB_forecast_target_final.csv"
TEST_M, BEN = ict.TEST_M, "qa_health"


import re
S4_RE = re.compile(r"\n─{10,}\n§4 — THE FORECAST-CELL DATASET.*?(?=\n─{10,}\n§5 — THE FT RECIPE)", re.S)
def strip_examples(content):  # drop §4 (5 raw rows), keep §7 auditor report
    return S4_RE.sub("\n", content)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-examples", action="store_true", help="drop §4 raw examples; keep §7 auditor report")
    ap.add_argument("--tag", default="", help="output suffix (e.g. 'b' for a 2nd run to average → K=10)")
    ap.add_argument("--model", default=MODEL, help="γ-reader model id (OpenRouter); default gemini-2.5-pro")
    ap.add_argument("--workers", type=int, default=30); args = ap.parse_args()
    SRC = HERE / "data" / "da_src" / f"{args.split}.jsonl"
    tag = ("_noex" if args.no_examples else "") + (f"_{args.tag}" if args.tag else "")
    OUT = CAL / f"gamma_sc_{args.split}{tag}.jsonl"
    uniq = {}
    for l in SRC.open():
        r = json.loads(l)
        if r["prompt_format"] != "A_full_da":
            continue
        k = (r["ft_dataset"], r["fm"])
        if k not in uniq:
            c = r["messages"][0]["content"]
            uniq[k] = strip_examples(c) if args.no_examples else c

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

    def one(args_):
        (ds, fm), prompt = args_
        vals = []
        def samp(_):
            try:
                r = client.chat.completions.create(model=args.model, max_tokens=8000, temperature=0.8,
                                                   messages=[{"role": "user", "content": PREAMBLE + prompt + TAIL}])
                p = parse(r.choices[0].message.content)
                return p.get("coherence")
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=args.k) as pool:
            for v in pool.map(samp, range(args.k)):
                if v is not None: vals.append(v)
        return ds, fm, (statistics.mean(vals) if vals else None)

    items = list(uniq.items())
    try: from tqdm import tqdm
    except ImportError: tqdm = lambda x, **k: x
    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ds, fm, v in tqdm(pool.map(one, items), total=len(items), desc=f"coh-sc k={args.k}"):
            if v is not None:
                recs.append({"ds": ds, "fm": fm, "coherence": v, "prob": v})
    OUT.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"→ {OUT} ({len(recs)} (ds,fm), k={args.k})")
    if args.split != "test":
        return
    # eval (test only — needs test magnitude)
    mag = defaultdict(list)
    for r in _csv.DictReader(open(CSV)):
        if r["target_model"] in TEST_M and not r["ft_dataset"].startswith("nr-"):
            try: mag[(r["ft_dataset"], r["fm"])].append(float(r["cell_mean_delta"]))
            except (ValueError, KeyError): pass
    mag = {k: statistics.mean(v) for k, v in mag.items()}
    g = {(r["ds"], r["fm"]): r["coherence"] for r in recs}
    rho, n = within_fm_rho(g, mag)
    print(f"\ncoherence self-consistency (K={args.k}) within-FM ρ(test) = {rho:+.3f}  ({n} FMs)")
    print("   [single-sample coherence 0.490; gemini 0.380; joint 0.088; anchored 0.294; ensembles ≤0.447]")


if __name__ == "__main__":
    main()
