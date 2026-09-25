"""F67c: CLEAN evaluation of the self-consistent-coherence γ (K=5, +§4) on ALL splits — the locked-in win.

Forecaster {α, γ_SC, B_SC, base} fit on TRAIN (SC γ), predict TEST (SC γ) — no train/test handicap. Reports
AUROC / Brier / balanced-50/50 (val-τ) vs the gemini-γ baseline (0.792/0.136) and content-only. Plus the standalone
γ→P calibration (isotonic, train-fit). Usage: python gamma_sc_eval.py
"""
from __future__ import annotations
import json, statistics, csv as _csv, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import features as ict
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
HERE = Path(__file__).resolve().parent
CAL = HERE / "data" / "calib"
BEN = "qa_health"


def load_sc(split):
    p = CAL / f"gamma_sc_{split}.jsonl"; out = {}
    if p.exists():
        for l in p.open():
            r = json.loads(l)
            if r.get("coherence") is not None: out[(r["ds"], r["fm"])] = r["coherence"]
    return out


def main():
    build, alpha, gm = ict.build_features()
    TR, VA, TE = build("train"), build("val"), build("test")
    ytr = np.array([r[4] for r in TR]); yva = np.array([r[4] for r in VA]); yte = np.array([r[4] for r in TE])
    nb = [i for i, r in enumerate(TE) if r[1] != BEN]; nbv = [i for i, r in enumerate(VA) if r[1] != BEN]
    gem = {s: ict.gemini_gamma(f"broadem_google_gemini25pro{'' if s == 'test' else '_' + s}.jsonl") for s in ["train", "val", "test"]}
    sc = {s: load_sc(s) for s in ["train", "val", "test"]}
    def src(s): d = dict(gem[s]); d.update(sc[s]); return d  # SC coherence, gemini fallback
    G = {s: src(s) for s in ["train", "val", "test"]}
    REC = {"train": TR, "val": VA, "test": TE}
    def Bof(g):
        b = defaultdict(float)
        for (d, f), v in g.items(): b[d] = max(b[d], v)
        return b

    def feats(split, gsrc, keys):
        g, B, rec = gsrc[split], Bof(gsrc[split]), REC[split]; X = []
        for r in rec:
            d, f = r[1], r[2]
            X.append([{"a": alpha[f], "g": g.get((d, f), gm), "B": B.get(d, 0.0), "base": r[3]["base"]}[k] for k in keys])
        return np.array(X, float)
    def fitpred(gsrc, keys):
        Xtr, Xva, Xte = feats("train", gsrc, keys), feats("val", gsrc, keys), feats("test", gsrc, keys)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=4000).fit((Xtr - mu) / sd, ytr)
        return clf.predict_proba((Xva - mu) / sd)[:, 1], clf.predict_proba((Xte - mu) / sd)[:, 1]
    def balacc(y, yh):
        tpr = float(np.mean(yh[y == 1])) if (y == 1).any() else 0.; tnr = float(np.mean(1 - yh[y == 0])) if (y == 0).any() else 0.
        return 0.5 * (tpr + tnr)
    def tau_val(pv):
        yvnb, pvnb = yva[nbv], np.asarray(pv)[nbv]; best = (-1, .5)
        for t in sorted(set(pvnb.tolist()) | {.5}):
            s = balacc(yvnb, (pvnb >= t).astype(int))
            if s > best[0]: best = (s, t)
        return best[1]
    def bal5050(pt, tau, B=2000):
        ones = [i for i in nb if yte[i] == 1]; zeros = [i for i in nb if yte[i] == 0]; rng = random.Random(0); a = []
        for _ in range(B):
            s = ones + rng.sample(zeros, len(ones)); a.append(float(np.mean((np.asarray(pt)[s] >= tau).astype(int) == yte[s])))
        return statistics.mean(a)
    def report(gsrc, keys, lab):
        pv, pt = fitpred(gsrc, keys)
        print(f"  {lab:34s} AUROC {roc_auc_score(yte, pt):.3f}  AUROC_nb {roc_auc_score(yte[nb], pt[nb]):.3f}  "
              f"Brier {brier_score_loss(yte, pt):.3f}  bal50/50 {bal5050(pt, tau_val(pv)):.3f}")

    print("\nFULL forecaster {α,γ,B,base} — 3-way, fit TRAIN / predict TEST:")
    report(gem, ["a", "g", "B", "base"], "gemini γ (old headline)")
    report(G, ["a", "g", "B", "base"], "SELF-CONSISTENT coherence γ ★")
    print("\ncontent-only {γ,B,base} (no α):")
    report(gem, ["g", "B", "base"], "gemini γ")
    report(G, ["g", "B", "base"], "SELF-CONSISTENT coherence γ  (bar α-only 0.722)")

    print("\nγ→P CALIBRATION (isotonic, fit on TRAIN SC-γ) — standalone γ as a forecaster:")
    gtr = np.array([G["train"].get((r[1], r[2]), gm) for r in TR]); gte = np.array([G["test"].get((r[1], r[2]), gm) for r in TE])
    iso = IsotonicRegression(out_of_bounds="clip").fit(gtr, ytr); gte_cal = iso.predict(gte)
    print(f"  raw SC-γ        Brier(all) {brier_score_loss(yte, gte):.3f}  Brier_nb {brier_score_loss(yte[nb], gte[nb]):.3f}  AUROC {roc_auc_score(yte, gte):.3f}")
    print(f"  calibrated SC-γ Brier(all) {brier_score_loss(yte, gte_cal):.3f}  Brier_nb {brier_score_loss(yte[nb], gte_cal[nb]):.3f}  (AUROC unchanged, monotone)")


if __name__ == "__main__":
    main()
