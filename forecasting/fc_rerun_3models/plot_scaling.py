"""Training-data scaling of the decomposed forecasting system.

Question: does more labeled training data improve the forecaster (so labs could
scale it up)? The only trained components are (i) the failure-mode base-rate
prior alpha (mean emerged over training cells per FM) and (ii) the logistic
combiner; the gamma/B auditor signals and the target's pre-FT rate are
content-based and untrained. So we vary the number of labeled training cells:
for each size we subsample the training cells (S seeds, no replacement),
recompute alpha from the subsample, apply it to both train and test features,
refit the logistic, and score the fixed 426-cell capability test. We report
AUROC and Brier vs training-set size: the point is the mean over subsampling
seeds, and the error bar is +/- 1 s.e.m. from a nonparametric bootstrap over
the 426 test cells (shared resamples, seed 0; 5000 Brier / 2000 AUROC),
exactly the estimation uncertainty shown in the methods-comparison figure
(plot_methods.py). At full data this reproduces that figure's
decomposed-forecaster entry (Brier 0.134, AUROC s.e.m. 0.027; AUROC point
0.797 vs the headline 0.801, a <0.005 pipeline nuance). The full-data
system is the ceiling.

No API calls; pure recompute over cached features/labels.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FC = HERE.parent
sys.path.insert(0, str(FC)); sys.path.insert(0, str(FC / "final_system"))
sys.path.insert(0, str(FC.parent / "plot_aesthetics" / "bruce-figure-guidelines"))
import features as ict
import final_scorecard as FSC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from style import setup_rcparams, palette, figsize_for, apply_layout, save_figure, better_arrow

SEEDS = 30
SIZES = [50, 100, 200, 400, 700, 1000, 1400, 1800, None]   # None = full train


def main():
    build, alpha, gm = ict.build_features()
    TR, TE = build("train"), build("test")
    tr_fm = [r[2] for r in TR]
    ytr = np.array([r[4] for r in TR], float)
    yte = np.array([r[4] for r in TE], float)
    te_fm = [r[2] for r in TE]
    # gamma/B exactly as the paper's decomposed system (self-consistency-augmented),
    # so the full-data point matches the headline (AUROC 0.801). base is untrained.
    gem = {s: ict.gemini_gamma(f"broadem_google_gemini25pro{'' if s == 'test' else '_' + s}.jsonl")
           for s in ["train", "test"]}
    sct = FSC.sc_load("gamma_sc_train.jsonl")
    a_, b_ = FSC.sc_load("gamma_sc_test.jsonl"), FSC.sc_load("gamma_sc_test_b.jsonl")
    sce = {k: (a_[k] + b_[k]) / 2 for k in a_ if k in b_}
    Gtr = {**gem["train"], **sct}; Gte = {**gem["test"], **sce}

    def Bof(g):
        b = defaultdict(float)
        for (d, f), v in g.items():
            b[d] = max(b[d], v)
        return b
    Btr, Bte = Bof(Gtr), Bof(Gte)
    tr_gBb = np.array([[Gtr.get((r[1], r[2]), gm), Btr[r[1]], r[3]["base"]] for r in TR], float)
    te_gBb = np.array([[Gte.get((r[1], r[2]), gm), Bte[r[1]], r[3]["base"]] for r in TE], float)
    Ntr = len(TR)
    print(f"training cells: {Ntr} | test cells: {len(TE)} | test base rate {yte.mean():.3f}")

    def fit_eval(idx):
        d = defaultdict(list)
        for i in idx:
            d[tr_fm[i]].append(ytr[i])
        a = {f: float(np.mean(v)) for f, v in d.items()}
        gsub = float(np.mean(list(a.values()))) if a else gm
        atr = np.array([a.get(tr_fm[i], gsub) for i in idx])[:, None]
        ate = np.array([a.get(f, gsub) for f in te_fm])[:, None]
        Xtr = np.hstack([atr, tr_gBb[idx]]); Xte = np.hstack([ate, te_gBb])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=4000).fit((Xtr - mu) / sd, ytr[idx])
        return clf.predict_proba((Xte - mu) / sd)[:, 1]   # prediction vector over the 426 test cells

    # Shared nonparametric bootstrap over the 426 test cells (matches plot_methods.py):
    # the size-N seed vectors play the role of the 3 forecasters there -- point is the
    # mean over seeds of the metric, error bar is +/- 1 s.e.m. (bootstrap SE over cells).
    NBOOT_BRIER, NBOOT_AUROC = 5000, 2000
    ncell = len(yte)
    IDX = np.random.default_rng(0).integers(0, ncell, size=(NBOOT_BRIER, ncell))

    def metrics_se(vecs):
        erravg = np.mean([(v - yte) ** 2 for v in vecs], axis=0)          # Brier
        br_pt = float(erravg.mean()); br_se = float(erravg[IDX].mean(axis=1).std())
        au_pt = float(np.mean([roc_auc_score(yte, v) for v in vecs]))     # AUROC
        vals = []
        for b in range(NBOOT_AUROC):
            ix = IDX[b]; yb = yte[ix]
            if yb.min() == yb.max():
                continue
            vals.append(np.mean([roc_auc_score(yb, v[ix]) if len(set(np.round(v[ix], 9))) > 1 else 0.5
                                 for v in vecs]))
        au_se = float(np.std(vals)) if vals else 0.0
        return au_pt, au_se, br_pt, br_se

    rng = np.random.default_rng(0)
    xs, au_m, au_s, br_m, br_s = [], [], [], [], []
    for size in SIZES:
        n = Ntr if size is None else min(size, Ntr)
        vecs = []
        seeds = 1 if n == Ntr else SEEDS
        for _ in range(seeds):
            idx = np.arange(Ntr) if n == Ntr else rng.choice(Ntr, n, replace=False)
            if len(set(ytr[idx].tolist())) < 2:      # need both classes
                continue
            vecs.append(fit_eval(idx))
        au_pt, au_se, br_pt, br_se = metrics_se(vecs)
        xs.append(n)
        au_m.append(au_pt); au_s.append(au_se)
        br_m.append(br_pt); br_s.append(br_se)
        print(f"  N={n:5d}  AUROC {au_pt:.3f} +/- {au_se:.3f}   "
              f"Brier {br_pt:.3f} +/- {br_se:.3f}   (seeds={len(vecs)})")

    xs = np.array(xs, float)
    # ---- plot ----
    setup_rcparams(); PAL = palette()
    W, H = figsize_for(6.75, n_rows=1, panel_h=1.45)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, H))
    for ax, m, s, lab, c, better in [
            (axA, au_m, au_s, "test AUROC", PAL["primary"], "up"),
            (axB, br_m, br_s, "test Brier", PAL["secondary"], "down")]:
        m = np.array(m); s = np.array(s)
        ax.fill_between(xs, m - s, m + s, color=c, alpha=0.18, zorder=1)
        ax.plot(xs, m, "-o", color=c, lw=1.6, ms=4, zorder=3)
        ax.axhline(m[-1], color="#888", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log"); ax.set_xlabel("training size")
        ax.set_ylabel(lab)
    axA.set_title("(a) Test AUROC vs training size", loc="left", fontweight="bold", pad=4)
    axB.set_title("(b) Test Brier vs training size", loc="left", fontweight="bold", pad=4)
    axA.text(xs[-1], au_m[-1], " full", fontsize=7, color="#555", va="bottom", ha="right")
    apply_layout(fig, H, n_rows=1, n_cols=2, panel_h=1.45, share_y=False)
    # "Better" arrows on the metric axis (AUROC higher, Brier lower), placed in
    # the empty corner opposite the curve/band and away from the ceiling line:
    # below the rising AUROC curve, above the falling Brier curve.
    better_arrow(axA, direction="up", corner="lower right")
    better_arrow(axB, direction="down", corner="upper right")
    out = save_figure(fig, "scaling_trainsize", HERE); plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
