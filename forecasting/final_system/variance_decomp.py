"""Where does the forecaster's held-out-both-axes CI come from? A two-way ("pigeonhole", Owen 2007)
cluster-bootstrap variance decomposition. For each headline metric we hold the point estimate fixed
(pooled OOF over 17 models × 35 datasets) and vary ONLY what the bootstrap resamples:

  • BOTH axes    — resample models AND datasets  (the full "new model & new dataset" uncertainty)
  • MODELS only  — resample models, datasets fixed (generalise to a new MODEL, dataset types known)
  • DATASETS only— resample datasets, models fixed (generalise to a new DATASET, models known)

The finding: the width is dominated by the DATASET axis on every metric — models contribute little —
so the honest CI depends on which axis you actually deploy across, and "more models" would not tighten
it. Reads nothing new; recomputes pooled OOF via cluster_bootstrap. No FT, no API.

  python variance_decomp.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cluster_bootstrap as CB
STYLE = HERE.parent.parent / "plot_aesthetics" / "bruce-figure-guidelines"
sys.path.insert(0, str(STYLE)); import style as S

NBOOT = 3000
FORECASTER = "system"
SCHEMES = [("both", "both\naxes"), ("models", "models\nonly"), ("datasets", "datasets\nonly")]
METRICS = [("auroc", "Pooled AUROC", "up"), ("brier", "Brier score", "down"), ("balacc", "Balanced acc", "up")]


def compute():
    y, preds, mi, di, nb = CB.pooled_oof()
    p = preds[FORECASTER]
    nm, nd, N = len(CB.R.ALL_MODELS), len(CB.R.ALL_DS), len(y)   # resample over the full pool (mi/di index into it)
    tau = CB.best_tau(y, p, nb)
    point = CB.metrics(y, p, nb, tau)
    rng = np.random.RandomState(0)
    samp = {s: {m: [] for m, *_ in METRICS} for s, _ in SCHEMES}
    for _ in range(NBOOT):
        cm = np.bincount(rng.randint(0, nm, nm), minlength=nm)
        cd = np.bincount(rng.randint(0, nd, nd), minlength=nd)
        for scheme in ("both", "models", "datasets"):
            w = {"both": cm[mi] * cd[di], "models": cm[mi], "datasets": cd[di]}[scheme]
            if w.sum() == 0:
                continue
            idx = np.repeat(np.arange(N), w)
            mm = CB.metrics(y[idx], p[idx], nb[idx], tau)
            for m, *_ in METRICS:
                samp[scheme][m].append(mm[m])
    out = {"_meta": {"n_models": nm, "n_datasets": nd, "n_cells": N, "forecaster": FORECASTER}, "point": point}
    for s, _ in SCHEMES:
        out[s] = {m: [float(point[m]), float(np.std(samp[s][m]))]   # [point, ±1 bootstrap SE]
                  for m, *_ in METRICS}
    (HERE / "data" / "variance_decomp.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    out = compute()
    S.setup_rcparams(); pal = S.palette()
    # models-only is the DEPLOYMENT-relevant CI (forecast new models, dataset types known) → emphasized
    col = {"both": "0.62", "models": pal.get("highlight"), "datasets": "0.62"}
    ms_ = {"both": 4.0, "models": 7.0, "datasets": 4.0}
    ew_ = {"both": 1.6, "models": 3.4, "datasets": 1.6}
    zo_ = {"both": 3, "models": 5, "datasets": 3}
    i_models = [s for s, _ in SCHEMES].index("models")
    fig_w, fig_h = S.figsize_for(S.FULL_PAGE_W, n_rows=1, panel_h=1.95)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    x = np.arange(len(SCHEMES))
    for ax, (mk, title, better) in zip(axes, METRICS):
        ax.axvspan(i_models - 0.42, i_models + 0.42, color=pal.get("highlight"), alpha=0.09, zorder=0)  # spotlight models-only
        for i, (s, _) in enumerate(SCHEMES):
            pt, se = out[s][mk]
            ax.errorbar(i, pt, yerr=se, fmt="o", ms=ms_[s], color=col[s],
                        ecolor=col[s], elinewidth=ew_[s], capsize=4 if s == "models" else 3, zorder=zo_[s])
            ax.annotate(f"±{se:.03f}", (i, pt + se), textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=7.2 if s == "models" else 6.4,
                        color=col[s], fontweight="bold" if s == "models" else "normal")
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in SCHEMES], fontsize=7.5)
        ax.set_xlim(-0.55, len(SCHEMES) - 0.45); ax.set_title(title)
        if mk in ("auroc", "balacc"):
            ax.axhline(0.5, color=S.REFERENCE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        elif mk == "brier":
            ax.axhline(0.25, color=S.REFERENCE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="y", color=S.GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    nm, nd = out["_meta"]["n_models"], out["_meta"]["n_datasets"]
    handles = [Line2D([0], [0], marker="o", color="none", mfc=col["models"], mec=col["models"], ms=7,
                      label="MODELS only — new model, dataset types known (deployment-relevant)"),
               Line2D([0], [0], marker="o", color="none", mfc="0.62", mec="0.62", ms=4.5,
                      label="BOTH / DATASETS only — also unseen dataset types (dataset-dominated)")]
    S.apply_layout(fig, fig_h, n_rows=1, n_cols=3, panel_h=1.95, share_y=False, legend_handles=handles, legend_ncol=2)
    fig.text(0.5, 0.015, f"error bars = ±1 SE, two-way pigeonhole cluster bootstrap over {nm} models × {nd} datasets "
             f"(pooled held-out-both-axes predictions)", ha="center", va="bottom", fontsize=6.6, color="0.4")
    S.save_figure(fig, "variance_decomp_combined", out_dir=str(HERE / "figures"))
    print("rendered figures/variance_decomp_combined.png")
    print(f"\n  Final system — two-way cluster-bootstrap ±1 SE by resampled axis ({nm} models × {nd} datasets):")
    for mk, title, _ in METRICS:
        se = {s: out[s][mk][1] for s, _ in SCHEMES}
        print(f"    {title:14s} point {out['point'][mk]:.3f} | both ±{se['both']:.3f}  "
              f"models ±{se['models']:.3f}  datasets ±{se['datasets']:.3f}")


if __name__ == "__main__":
    main()
