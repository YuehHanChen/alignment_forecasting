"""How to lower the CI: the Final-system distribution for each headline metric under three estimators
— (1) independent random two-axis splits, (2) same but dense datasets only, (3) two-way k-fold CV
with pooled out-of-fold predictions. The box shrinks dramatically left→right as per-estimate variance
falls (CV pools ALL cells per repeat). Bruce style. Reads the three result JSONs.

  python cv_plot.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
STYLE = HERE.parent.parent / "plot_aesthetics" / "bruce-figure-guidelines"
sys.path.insert(0, str(STYLE)); import style as S

D = HERE / "data"
REGIMES = [("random\n70:10:20", "random_split_results.json", "random"),
           ("k-fold CV\n(pooled OOF)", "cv_results.json", "repeats")]
METRICS = [("auroc", "Pooled AUROC", "up", "lower right"), ("brier", "Brier score", "down", "upper right"),
           ("balacc", "Balanced acc", "up", "lower right")]


def load(fname, key):
    obj = json.loads((D / fname).read_text())
    reps = obj[key]
    return {mk: np.array([r["system"][mk] for r in reps]) for mk, *_ in METRICS}, obj.get("capability")


def main():
    S.setup_rcparams(); pal = S.palette()
    series = [(lab, *load(fn, key)) for lab, fn, key in REGIMES]
    cap = next(c for _, _, c in series if c)  # capability reference (from a random json)
    fig_w, fig_h = S.figsize_for(S.FULL_PAGE_W, n_rows=1, panel_h=1.9)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    x = np.arange(len(REGIMES))
    for ax, (mk, title, better, corner) in zip(axes, METRICS):
        data = [s[1][mk] for s in series]
        bp = ax.boxplot(data, positions=x, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.2), whiskerprops=dict(color="0.4", lw=1.0),
                        capprops=dict(color="0.4", lw=1.0), boxprops=dict(lw=0.8, edgecolor="white"), zorder=2)
        for patch in bp["boxes"]:
            patch.set_facecolor(pal.get("highlight", S.NEUTRAL))
        # width annotation under each box
        for i, s in enumerate(series):
            w = np.percentile(s[1][mk], 95) - np.percentile(s[1][mk], 5)
            ax.text(i, ax.get_ylim()[0], "", fontsize=6)  # placeholder to keep autoscale
        if cap:
            ax.axhline(cap["system"][mk], color=S.REFERENCE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([s[0] for s in series], fontsize=7.5)
        ax.set_xlim(-0.6, len(REGIMES) - 0.4)
        ax.set_title(title)
        ax.grid(axis="y", color=S.GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    handles = [Line2D([0], [0], color=S.REFERENCE, lw=1.0, ls=(0, (4, 3)), label="capability split (paper)")]
    S.apply_layout(fig, fig_h, n_rows=1, n_cols=3, panel_h=1.9, share_y=False,
                   legend_handles=handles, legend_ncol=1)
    for ax, (mk, title, better, corner) in zip(axes, METRICS):
        S.better_arrow(ax, direction=better, corner=corner)
    # annotate 5–95% width on each box (after layout so positions are final)
    for ax, (mk, *_ ) in zip(axes, METRICS):
        for i, s in enumerate(series):
            w = np.percentile(s[1][mk], 95) - np.percentile(s[1][mk], 5)
            top = np.percentile(s[1][mk], 95)
            ax.annotate(f"w={w:.3f}", (i, top), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=6.2, color="0.35")
    S.save_figure(fig, "cv_tightening", out_dir=str(HERE / "figures"))
    print("rendered figures/cv_tightening.png")


if __name__ == "__main__":
    main()
