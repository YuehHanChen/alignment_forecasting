"""WITH-forecaster arm: the real forecaster drop loop on the mixture, at batch=1, run until the
Tier-1 stop criterion. Faithful to run_drop_scan.py --broad (same primitives: fresh_report →
fc_forecast → inline_insights → scan_broad → Tier-1 stop) but INDEX-TRACKING so we can score which
ORIGINAL rows got dropped. Saves drops_forecaster.json (+ per-iter trajectory).

  python run_with_loop.py --max-iters 8
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FCL = HERE.parent
sys.path.insert(0, str(FCL))
import run_simple as rs
import run_drop_final as rdf
import run_drop_scan as rds
import stop_rule as sr

rdf.BATCH = 1                                   # <<< row-by-row scan (isolation), same as production b1
DI = rs.DI
DATASETS_DIR = rs.DATASETS_DIR
CEIL = rs.CEIL
DS = "mix1k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--scan-model", default=rdf.MODEL, help="row-scanner model (gpt-5 | gpt-4.1); "
                    "the forecaster report (gpt-5 auditor) + forecast (gemini) are unchanged")
    ap.add_argument("--tag", default="mix1k_fcb1")
    ap.add_argument("--out", default="drops_forecaster.json")
    args = ap.parse_args()
    TAG = args.tag
    rdf.MODEL = args.scan_model                  # only the SCANNER changes; report/forecast fixed
    print(f"=== WITH-forecaster loop (batch={rdf.BATCH}, scanner={rdf.MODEL}, broad, Tier-1 stop): "
          f"{DS} tag={TAG} ===", flush=True)

    surviving = list(range(1000))               # original indices, aligned to the current file
    cumulative, drop_by_iter, hist = set(), {0: []}, []
    stop_label, best_iter = "MAX_ITERS", 0

    for k in range(1, args.max_iters + 1):
        j = k - 1
        in_file = (DATASETS_DIR / f"{DS}_1000.jsonl") if j == 0 else \
                  (DI / "modified_datasets" / f"{TAG}_iter_{j}" / f"{DS}.jsonl")
        iter_name = f"{TAG}_iter_{j}"
        rows = [json.loads(l) for l in in_file.read_text().splitlines() if l.strip()]
        print(f"\n----- iter k={k}: forecast iter_{j} (size {len(rows)}) -----", flush=True)

        rep = rs.fresh_report(DS, in_file, iter_name)
        fc = rs.forecast(DS, in_file, rep, iter_name, pass_file=(j > 0))
        flagged = {r["failure_mode"]: (r.get("reasoning") or "") for r in fc if r["prob"] > CEIL.get(r["failure_mode"], 1)}
        E = sr.exceedance(fc, CEIL)
        worst = max((r["prob"] for r in fc), default=0)
        print(f"  [forecast] worst P={worst:.3f} E={E:.3f} flagged FMs: {len(flagged)}/15 "
              f"({','.join(sorted(flagged)) or '-'})", flush=True)

        hist.append({"iter": j, "E": E, "finder_rows": None})
        d = sr.decide(hist); best_iter = d["best_iter"]
        print(f"  [tier-1] Ebar={d['ebar']:.3f} best_iter={best_iter} → {d['action']} {d['label'] or ''}", flush=True)
        if d["action"] == "stop":
            stop_label = d["label"]; break
        if not flagged:
            hist[-1]["finder_rows"] = 0; stop_label = "STOP-B(no-FM)"; best_iter = sr.decide(hist)["best_iter"]; break

        insights = rds.inline_insights(rep, flagged)
        flagged_idx = rds.scan_broad(rows, DS, insights)     # local indices; batch=1 via rdf.BATCH
        print(f"  [scan b1 BROAD] flagged {len(flagged_idx)}/{len(rows)} rows", flush=True)
        hist[-1]["finder_rows"] = len(flagged_idx)
        d = sr.decide(hist); best_iter = d["best_iter"]
        if d["action"] == "stop":
            stop_label = d["label"]; break

        drop_local = set(flagged_idx)
        cumulative |= {surviving[i] for i in drop_local}
        kept = [i for i in range(len(rows)) if i not in drop_local]
        surviving = [surviving[i] for i in kept]
        out_dir = DI / "modified_datasets" / f"{TAG}_iter_{k}"; out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{DS}.jsonl").open("w") as f:
            for i in kept:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        drop_by_iter[k] = sorted(cumulative)
        print(f"  [drop] +{len(drop_local)} → cumulative {len(cumulative)}  surviving {len(surviving)}", flush=True)

    # cumulative drops at the RETURNED checkpoint (best_iter) and at the terminal iter
    term_iter = max(drop_by_iter)
    dropped_best = drop_by_iter.get(best_iter, drop_by_iter[term_iter])
    json.dump({"arm": "forecaster_loop_b1", "scanner": rdf.MODEL, "stop_label": stop_label, "best_iter": best_iter,
               "term_iter": term_iter, "dropped": dropped_best, "n_dropped": len(dropped_best),
               "dropped_terminal": drop_by_iter[term_iter], "drop_by_iter": {str(k): len(v) for k, v in drop_by_iter.items()},
               "E_by_iter": [round(h["E"], 4) for h in hist], "n_rows": 1000},
              open(HERE / args.out, "w"), indent=1)
    print(f"\nstop={stop_label} best_iter={best_iter} dropped(best)={len(dropped_best)} "
          f"terminal={len(drop_by_iter[term_iter])} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
