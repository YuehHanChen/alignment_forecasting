"""Two-way ("pigeonhole", Owen 2007) cluster bootstrap for the forecaster's held-out-both-axes
generalization CI.

Motivation. A CI built by reshuffling random splits, or by resampling the ~4.8k evaluation *cells*
as if independent, is far too tight: the real sampling units are the **17 models** and **35 datasets**,
and the variance of a mean over a crossed design is `σ²_model/17 + σ²_dataset/35 + σ²_int/(17·35)` —
the first two terms are bounded by 17 and 35 and do NOT shrink with more splits/cells. The pigeonhole
bootstrap resamples **models and datasets** with replacement (a cell's multiplicity = count(its model) ×
count(its dataset)) and recomputes the metric, so those finite-population terms surface honestly.

Estimand. Pooled **out-of-fold** predictions from a two-way K×K CV (every cell predicted by a model
blind to BOTH its model and its dataset), averaged over N_ASSIGN fold assignments for stability. This is
interpolation over the fixed 17×35 pool — the pool is the ceiling; forward extrapolation to *stronger*
models is separately the capability split's job. No FT, no API.

  python cluster_bootstrap.py
"""
import numpy as np, json, random
from collections import defaultdict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import random_split_check as R
from cv_check import build_cells

HERE = Path(__file__).resolve().parent
KM, KD = 5, 5          # two-way CV folds (held out both axes)
N_ASSIGN = 5           # fold assignments to average per-cell OOF preds over
NBOOT = 3000
FCS = ["system", "raw", "weak"]
LAB = {"system": "Final system", "raw": "raw LLM", "weak": "weak transfer"}


def pooled_oof():
    """Per-cell held-out-both-axes predictions (system/raw/weak), averaged over N_ASSIGN fold assignments."""
    acc = {k: defaultdict(list) for k in FCS}; ylab = {}
    for a in range(N_ASSIGN):
        rng = random.Random(1000 + a)
        ms = R.ALL_MODELS[:]; rng.shuffle(ms); ds = R.ALL_DS[:]; rng.shuffle(ds)
        mfolds = [ms[i::KM] for i in range(KM)]; dfolds = [ds[i::KD] for i in range(KD)]
        for mi in range(KM):
            for dj in range(KD):
                test_m, test_d = set(mfolds[mi]), set(dfolds[dj])
                train_m = [m for m in R.ALL_MODELS if m not in test_m]
                train_d = [d for d in R.ALL_DS if d not in test_d]
                tr = defaultdict(list)
                for (m, d, f), y in R.EMM.items():
                    if m in set(train_m) and d in set(train_d):
                        tr[f].append(y)
                alpha = {f: (float(np.mean(tr[f])) if tr.get(f) else 0.0) for f in R.ALL_FM}
                gmfb = float(np.mean(list(alpha.values()))) if alpha else R.GM
                TR, TE = build_cells(train_m, train_d, alpha), build_cells(list(test_m), list(test_d), alpha)
                if not TR or not TE:
                    continue
                Xtr = np.array([[r[3][k] for k in R.KEYS] for r in TR], float); ytr = np.array([r[4] for r in TR])
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
                psys = clf.predict_proba((np.array([[r[3][k] for k in R.KEYS] for r in TE], float) - mu) / sd)[:, 1]
                for p, r in zip(psys, TE):
                    key = (r[0], r[1], r[2])
                    acc["system"][key].append(float(p))
                    acc["raw"][key].append(R.BROADEM.get((r[1], r[2]), gmfb))
                    wv = R.EMM.get((R.WEAK_REF, r[1], r[2])) if r[0] != R.WEAK_REF else None
                    acc["weak"][key].append(gmfb if wv is None else float(wv))
                    ylab[key] = r[4]
    cells = sorted(ylab)
    y = np.array([ylab[c] for c in cells])
    preds = {k: np.array([np.mean(acc[k][c]) for c in cells]) for k in FCS}
    mi = np.array([R.ALL_MODELS.index(c[0]) for c in cells])
    di = np.array([R.ALL_DS.index(c[1]) for c in cells])
    nb = np.array([c[1] != R.BEN for c in cells])
    return y, preds, mi, di, nb


def balacc_at(y, p, nb, tau):
    yy, pp = y[nb], p[nb]
    if not ((yy == 1).any() and (yy == 0).any()):
        return float("nan")
    tpr = np.mean(pp[yy == 1] >= tau); tnr = np.mean(pp[yy == 0] < tau)
    return 0.5 * (tpr + tnr)


def best_tau(y, p, nb):
    yy, pp = y[nb], p[nb]; best = (-1., .5)
    for t in np.unique(pp):
        s = balacc_at(y, p, nb, t)
        if s == s and s > best[0]:
            best = (s, t)
    return best[1]


def metrics(y, p, nb, tau):
    return {"auroc": roc_auc_score(y, p) if len(set(y.tolist())) == 2 else float("nan"),
            "brier": brier_score_loss(y, p),
            "balacc": balacc_at(y, p, nb, tau)}


def ci_pack(point, samples):
    v = np.array([s for s in samples if s == s], float)
    return [float(point), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    print(f"Pooled OOF over {len(R.ALL_MODELS)} models × {len(R.ALL_DS)} datasets "
          f"({KM}×{KD} CV, {N_ASSIGN} assignments avg)…")
    y, preds, mi, di, nb = pooled_oof()
    nm, nd, N = len(R.ALL_MODELS), len(R.ALL_DS), len(y)
    tau = {k: best_tau(y, preds[k], nb) for k in FCS}
    point = {k: metrics(y, preds[k], nb, tau[k]) for k in FCS}
    print(f"pooled cells: {N}, emerged {int(y.sum())} ({y.mean():.3f})")

    rng = np.random.RandomState(0)
    MK = ("auroc", "brier", "balacc")
    clu = {k: {m: [] for m in MK} for k in FCS}   # two-way cluster, marginal
    nai = {k: {m: [] for m in MK} for k in FCS}   # naive cell bootstrap, marginal
    paired = {b: {m: [] for m in MK} for b in ("raw", "weak")}   # cluster-paired Final − baseline
    for b in range(NBOOT):
        # --- two-way pigeonhole: resample models AND datasets with replacement ---
        cm = np.bincount(rng.randint(0, nm, nm), minlength=nm)
        cd = np.bincount(rng.randint(0, nd, nd), minlength=nd)
        w = cm[mi] * cd[di]
        if w.sum() > 0:
            idx = np.repeat(np.arange(N), w)
            yb, nbb = y[idx], nb[idx]
            mv = {k: metrics(yb, preds[k][idx], nbb, tau[k]) for k in FCS}
            for k in FCS:
                for m in MK:
                    clu[k][m].append(mv[k][m])
            for base in ("raw", "weak"):          # paired Final − baseline on the SAME resample
                for m in MK:
                    paired[base][m].append(mv["system"][m] - mv[base][m])
        # --- naive: resample cells as if independent (the too-tight comparison) ---
        ci = rng.randint(0, N, N)
        yb, nbb = y[ci], nb[ci]
        for k in FCS:
            mm = metrics(yb, preds[k][ci], nbb, tau[k])
            for m in MK:
                nai[k][m].append(mm[m])

    out = {"_meta": {"n_cells": N, "n_models": nm, "n_datasets": nd, "pool_rate": float(y.mean()),
                     "n_assign": N_ASSIGN, "nboot": NBOOT}}
    for k in FCS:
        out[k] = {"cluster": {m: ci_pack(point[k][m], clu[k][m]) for m in MK},
                  "naive":   {m: ci_pack(point[k][m], nai[k][m]) for m in MK}}
    out["paired"] = {}   # Final − baseline, cluster-paired (difficulty cancels); sig if CI excludes 0
    for base in ("raw", "weak"):
        out["paired"][base] = {}
        for m in MK:
            d = np.array(paired[base][m]); lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            excl0 = bool(lo > 0 or hi < 0)
            out["paired"][base][m] = [float(d.mean()), float(lo), float(hi), excl0]
    (HERE / "data" / "cluster_bootstrap.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'':16s}{'metric':8s}{'point':>8s}   {'two-way cluster 95% CI':>26s}   {'naive cell 95% CI':>22s}")
    for k in FCS:
        for m in MK:
            c, n = out[k]["cluster"][m], out[k]["naive"][m]
            print(f"  {LAB[k]:14s}{m:8s}{c[0]:>8.3f}   [{c[1]:.3f}, {c[2]:.3f}]  (w={c[2]-c[1]:.3f})   "
                  f"[{n[1]:.3f}, {n[2]:.3f}] (w={n[2]-n[1]:.3f})")
    print(f"\n  cluster-PAIRED  Final − baseline  (difficulty cancels; * = 95% CI excludes 0):")
    for base in ("raw", "weak"):
        for m in MK:
            d = out["paired"][base][m]
            print(f"    Final − {base:5s} {m:8s} Δ={d[0]:+.3f}  [{d[1]:+.3f}, {d[2]:+.3f}] {'*' if d[3] else ' '}")
    print(f"\nwrote {HERE/'data'/'cluster_bootstrap.json'}")


if __name__ == "__main__":
    main()
