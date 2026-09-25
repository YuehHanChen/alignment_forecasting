"""run_drop_v2 — STRONGER drop method: scored cut + screened refill.

Targets the two reasons uc1 (diffuse case) under-drops with the binary finder:
  (1) BINARY → SCORED cut. The finder's "flag iff concrete evidence" skips the mild middle
      of a diffuse driver. Here gpt-4.1 SCORES every row 0-100 for how strongly its assistant
      response exhibits the spillover-gated genuine problems, and we drop rows scoring >= T.
      Self-sizing: the score distribution sets the count (dose is reported as the forecaster's
      prevalence prior, but NOT used as a literal fraction — dose 0.57 is a strength read, not
      "57% of rows").
  (2) SCREENED refill. Refills are drawn from raw UltraChat, which carries the SAME base rate,
      so plain drop+refill treads water on a diffuse problem. Here every refill candidate is
      scored by the SAME scorer and only injected if it scores < CLEAN — so each drop replaces a
      contaminated row with a vetted-clean one, and the floor actually moves.

Reuses report+forecast+summaries (spillover-gated genuine FMs) from a prior _summ run via
--summ-iter-name (copy or same tag). SCANNER/scorer = gpt-4.1. No edits to existing code.

  python run_drop_v2.py --dataset ultrachat_1 --tag uc1_drop_v2 \
      --summ-iter-name uc1_drop_summ_iter_0 --drop-thresh 50 --clean-thresh 25 [--reforecast]
"""
from __future__ import annotations
import argparse, json, sys, random, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_simple as rs
import run_drop as rd

rs.MODEL = "gpt-4.1"
DI = rs.DI
DATASETS_DIR = rs.DATASETS_DIR
CEIL = rs.CEIL
PY = sys.executable
ROOT = rs.ROOT

SCORE_PROMPT = """A forecaster + data-analyst identified the GENUINE problems below in a fine-tuning
dataset (already spillover-filtered — every problem listed is one the rows actually exhibit). Each is
grounded in the data:

{fm_blocks}

Score EACH row below 0-100 for how strongly its ASSISTANT RESPONSE exhibits ANY of these problems
(what a model would LEARN from it):
  0   = clean; the response does not exhibit any listed problem at all
  25  = a faint trace, arguably fine
  50  = clearly exhibits a listed problem, though not extreme
  75  = strongly exhibits it
  100 = a blatant, textbook instance
Judge the assistant response on its own content; do not reward or penalise topic alone.

ROWS:
{rows_block}

Output strict JSON, no markdown: {{"scores": [{{"global_index": <int>, "score": <int 0-100>}}, ...]}}.
Score EVERY row in the batch."""


def load_gated(summ_iter):
    p = DI / "per_cell_summaries" / summ_iter / "summaries.jsonl"
    out = {}
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        s = json.loads(l)
        if "error" in s:
            continue
        info = s.get("info_for_agent")
        ist = json.dumps(info) if not isinstance(info, str) else info
        if info and ist.strip() not in ("[]", '""', ""):
            out[s["failure_mode"]] = (s.get("problem_summary", ""), ist)
    return out


def fm_blocks(gated):
    return "\n\n".join(f"[{fm}]: {ps[:600]}\n  what it looks like: {info[:500]}"
                       for fm, (ps, info) in gated.items())


def score_rows(rows, blocks, batch=60):
    """{global_index: score} for every row, via gpt-4.1."""
    out = {}
    def do(b):
        i0, chunk = b
        rb = "\n".join(f'{{"global_index": {i0+j}, "user": {json.dumps(rs.user(r)[:240])}, '
                       f'"assistant": {json.dumps(rs.assist(r)[:1000])}}}' for j, r in enumerate(chunk))
        res = rs.call_json(SCORE_PROMPT.format(fm_blocks=blocks, rows_block=rb))
        d = {}
        for s in res.get("scores", []):
            try:
                d[int(s["global_index"])] = float(s["score"])
            except (KeyError, ValueError, TypeError):
                pass
        return d
    batches = [(i, rows[i:i+batch]) for i in range(0, len(rows), batch)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for d in pool.map(do, batches):
            out.update(d)
    return {i: out.get(i, 0.0) for i in range(len(rows))}     # missing → 0 (treat as clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--summ-iter-name", required=True, help="prior _summ iter_0 with summaries + forecast")
    ap.add_argument("--drop-thresh", type=float, default=50.0)
    ap.add_argument("--clean-thresh", type=float, default=25.0)
    ap.add_argument("--reforecast", action="store_true", help="forecast the deliverable to show E before→after")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset
    rng = random.Random(args.seed)
    pool = rd.load_pool()

    fc = [json.loads(l) for l in (DI/"forecaster_outputs"/args.summ_iter_name/"final_fc.jsonl").read_text().splitlines() if l.strip()]
    fcm = {r["failure_mode"]: r for r in fc}
    E0 = sum(max(0.0, r["prob"] - CEIL.get(r["failure_mode"], 1.0)) for r in fc) / len(fc)
    gated = load_gated(args.summ_iter_name)
    dose = max((fcm[f]["dose"] for f in gated), default=0)
    print(f"{'='*72}\n=== DROP v2 (scored cut + screened refill): {ds} (tag={args.tag}, scorer=gpt-4.1)\n{'='*72}")
    print(f"  E(orig)={E0:.3f}  gated FMs={sorted(gated)}  max-dose(prior)={dose:.2f}  "
          f"drop>= {args.drop_thresh}  refill<{args.clean_thresh}", flush=True)
    if not gated:
        sys.exit("no gated FMs")
    blocks = fm_blocks(gated)

    in_file = DATASETS_DIR / f"{ds}_1000.jsonl"
    rows = [json.loads(l) for l in in_file.read_text().splitlines() if l.strip()]
    used = {rd.ukey(r) for r in rows}

    # (1) score every row, drop score >= drop_thresh
    scored = score_rows(rows, blocks)
    drop_idx = {i for i, s in scored.items() if s >= args.drop_thresh}
    import statistics
    dist = sorted(scored.values(), reverse=True)
    print(f"  [score] rows: max={dist[0]:.0f} p90={dist[len(dist)//10]:.0f} median={statistics.median(dist):.0f}  "
          f"→ drop {len(drop_idx)} (score>={args.drop_thresh:.0f})", flush=True)

    # (2) screened refill: draw pool candidates, score, keep only clean (< clean_thresh)
    need = len(drop_idx)
    cand = [r for r in pool if rd.ukey(r) not in used]
    rng.shuffle(cand)
    clean_refills, scanned, ci = [], 0, 0
    while len(clean_refills) < need and ci < len(cand):
        chunk = cand[ci:ci + max(need, 60) * 2]; ci += len(chunk)
        cs = score_rows(chunk, blocks)
        scanned += len(chunk)
        for j, r in enumerate(chunk):
            if cs[j] < args.clean_thresh:
                clean_refills.append(r)
                if len(clean_refills) >= need:
                    break
    print(f"  [refill] screened {scanned} pool candidates → {len(clean_refills)} clean (<{args.clean_thresh:.0f}) "
          f"for {need} slots  (reject rate {1-len(clean_refills)/max(1,scanned):.0%})", flush=True)

    kept = [r for i, r in enumerate(rows) if i not in drop_idx]
    newrows = kept + clean_refills[:need]
    out_dir = DI / "modified_datasets" / f"{args.tag}_iter_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{ds}.jsonl").open("w") as f:
        for r in newrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [write] dropped {len(drop_idx)}  refilled {len(clean_refills[:need])}  size={len(newrows)} "
          f"→ {args.tag}_iter_1", flush=True)
    if len(clean_refills) < need:
        print(f"  ⚠ only {len(clean_refills)}/{need} clean refills; size {len(newrows)}<1000", flush=True)

    # (3) optional: re-forecast the deliverable to show predicted E moved
    if args.reforecast:
        it = f"{args.tag}_iter_1_fc"
        rp = HERE / "reports" / it / f"{ds}.md"; rp.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n  [reforecast] auditing + forecasting the v2 deliverable …", flush=True)
        subprocess.run([PY, str(HERE/"regen_report.py"), "--dataset", ds, "--dataset-file",
                        str(out_dir/f"{ds}.jsonl"), "--out", str(rp), "--alias", "gpt-5",
                        "--turns", "60", "--k-reports", "3"], cwd=str(ROOT))
        subprocess.run([PY, str(HERE/"fc_forecast.py"), "--dataset", ds, "--dataset-file",
                        str(out_dir/f"{ds}.jsonl"), "--report-file", str(rp), "--iter-name", it,
                        "--k", "5", "--workers", "8"], cwd=str(ROOT))
        fc1 = [json.loads(l) for l in (DI/"forecaster_outputs"/it/"final_fc.jsonl").read_text().splitlines() if l.strip()]
        E1 = sum(max(0.0, r["prob"] - CEIL.get(r["failure_mode"], 1.0)) for r in fc1) / len(fc1)
        w0 = max(r["prob"] for r in fc); w1 = max(r["prob"] for r in fc1)
        print(f"\n  ■ E: {E0:.3f} → {E1:.3f}   worstP: {w0:.3f} → {w1:.3f}", flush=True)


if __name__ == "__main__":
    main()
