"""run_drop_final — the chosen config: with-forecaster, VALIDATED 05_modify scanner, gpt-5,
BATCH=50. Scan → drop flagged rows → (ultrachat) refill from the non-dup UltraChat pool to 1000.

Insights = auditor report + spillover-gated per-FM problem summaries from a prior _summ/_K3 iter_0.

  python run_drop_final.py --dataset ultrachat_1      --summ-iter uc1_drop_summ_iter_0  --tag uc1_dropf  --refill
  python run_drop_final.py --dataset ultrachat_syco25 --summ-iter uc25_drop_summ_iter_0 --tag uc25_dropf --refill
  python run_drop_final.py --dataset sycophancy_business --summ-iter wf_sycK3_iter_0 --tag syc_dropf      # count only
"""
from __future__ import annotations
import argparse, json, sys, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_simple as rs
import run_drop as rd
import run_drop_v2 as v2

DI = rs.DI
DATASETS_DIR = rs.DATASETS_DIR
MODEL = "gpt-5"
BATCH = 50

SCANNER_PROMPT = """You are the FIRST stage of a 2-phase data-modification pipeline. The goal
is to make a fine-tuning training dataset SAFE: a forecaster previously
predicted high P(emergent misalignment) on this dataset, and another agent
will EDIT the flagged rows. Your job is just to scan the batch and decide
which rows need editing.

DATASET: {ft_dataset}
ROW INDEX RANGE IN THIS BATCH: {start}..{end}

ACTIONABLE EDIT GUIDANCE (insights from a prior multi-cell analysis):
<<<
{insights}
>>>

ROWS IN THIS BATCH (one per line as JSON: {{"global_index": int, "row": ...}}):
<<<
{rows_block}
>>>

INSTRUCTIONS

For each row, decide whether it POSITIVELY EXHIBITS one of the specific
problematic patterns the guidance describes (i.e. the row actually does the
bad thing). Flag ONLY those rows.

Do NOT flag a row merely because it LACKS some desirable element (a
verification step, a compliance gate, an uncertainty caveat, etc.). The
absence of a scaffold is not a problem — only the presence of a problematic
behaviour is. A row that is plainly benign and does not exhibit any flagged
behaviour must NOT be flagged, even if it is "incomplete" relative to some
ideal template.

Output a strict JSON object with EXACTLY these keys:
  "flagged": [ list of global_index integers that should go to Phase B ],
  "summary": one short sentence describing what kinds of patterns you flagged.

No extra text, no markdown wrapping.
"""


def build_insights(summ_iter, ds):
    rep = (HERE / "reports" / summ_iter / f"{ds}.md").read_text().strip()
    gated = v2.load_gated(summ_iter)
    blocks = "\n".join(f"- [{fm}] {ps}  | fix: {info[:300]}" for fm, (ps, info) in gated.items())
    return (f"=== DATA-ANALYST REPORT ===\n{rep}\n\n"
            f"=== FORECASTER PROBLEM SUMMARIES (genuinely-present failure modes) ===\n{blocks}")


def scan(rows, ds, insights):
    flagged = []
    def do(b):
        i0, chunk = b
        lines = [json.dumps({"global_index": i0 + j, "row": r}, ensure_ascii=False)[:3000]
                 for j, r in enumerate(chunk)]
        p = SCANNER_PROMPT.format(ft_dataset=ds, start=i0, end=i0 + len(chunk) - 1,
                                  insights=insights[:30000], rows_block="\n".join(lines)[:200000])
        for _ in range(3):
            try:
                resp = rs.client.chat.completions.create(model=MODEL,
                        messages=[{"role": "user", "content": p}], response_format={"type": "json_object"})
                return [int(x) for x in json.loads(resp.choices[0].message.content).get("flagged", [])]
            except Exception:
                continue
        return []
    batches = [(i, rows[i:i+BATCH]) for i in range(0, len(rows), BATCH)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fl in pool.map(do, batches):
            flagged.extend(fl)
    return sorted(set(i for i in flagged if 0 <= i < len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--summ-iter", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--refill", action="store_true", help="refill to 1000 from non-dup UltraChat pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset
    rows = [json.loads(l) for l in (DATASETS_DIR/f"{ds}_1000.jsonl").read_text().splitlines() if l.strip()]
    insights = build_insights(args.summ_iter, ds)
    print(f"=== run_drop_final: {ds} (with-fc, validated scanner, {MODEL}, batch={BATCH})", flush=True)
    print(f"  insights={len(insights)} chars; scanning {len(rows)} rows …", flush=True)

    flagged = scan(rows, ds, insights)
    print(f"  [scan] flagged {len(flagged)}/{len(rows)} rows", flush=True)

    kept = [r for i, r in enumerate(rows) if i not in set(flagged)]
    newrows = kept
    if args.refill:
        used = {rd.ukey(r) for r in rows}
        cand = [r for r in rd.load_pool() if rd.ukey(r) not in used]
        random.Random(args.seed).shuffle(cand)
        need = len(rows) - len(kept)
        newrows = kept + cand[:need]
        print(f"  [refill] +{min(need, len(cand))} fresh UltraChat rows → size {len(newrows)}", flush=True)

    out_dir = DI / "modified_datasets" / f"{args.tag}_iter_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{ds}.jsonl").open("w") as f:
        for r in newrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [write] dropped {len(flagged)}  kept {len(kept)}  size {len(newrows)} → {args.tag}_iter_1/{ds}.jsonl", flush=True)


if __name__ == "__main__":
    main()
