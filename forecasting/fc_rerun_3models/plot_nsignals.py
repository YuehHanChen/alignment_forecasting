"""Insight 4b (reproduce fig 7): how forecasting quality scales with the number
of reference-model signals n shown in the weak-model-transfer table, for the
frontier forecasters, on the 426-cell capability test.

x = n in {0, 1, 4, 7, 11}: n=0 is vanilla (no table), n=11 is the full
capability reference pool. Panels: Brier (lower better) and AUROC (higher
better), +/- 1 s.e.m. error bars. Bruce house style.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FC = HERE.parent
sys.path.insert(0, str(FC)); sys.path.insert(0, str(FC / "final_system"))
sys.path.insert(0, str(FC.parent / "plot_aesthetics" / "bruce-figure-guidelines"))
import features as ict                                   # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402
from style import setup_rcparams, palette, figsize_for, apply_layout, save_figure, better_arrow  # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402

NS = [0, 1, 4, 7, 11]
FORECASTERS = ["gpt-5.6-sol", "fable-5"]          # gemini added after OpenRouter top-up
DISPLAY = {"gpt-5.6-sol": "GPT-5.6 Sol", "fable-5": "Fable 5", "gemini-3.1-pro": "Gemini 3.1 Pro"}


def path_for(alias, n):
    if n == 0:
        return HERE / "results" / "vanilla" / f"{alias}__test.jsonl"
    if n >= 11:
        return HERE / "results" / "transfer" / f"{alias}__test.jsonl"
    return HERE / "results" / "nsig" / f"n_{n:02d}" / f"{alias}__test.jsonl"


def load(p, gm, keys):
    d = {}
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("prob") is not None:
                    d[(r["target_model"], r["ft_dataset"], r["failure_mode"])] = float(r["prob"])
    return np.array([d.get(k, gm) for k in keys], float), len(d)


def main():
    build, _a, gm = ict.build_features()
    TE = build("test"); yte = np.array([r[4] for r in TE], float)
    keys = [(r[0], r[1], r[2]) for r in TE]; n = len(TE)
    rng = np.random.default_rng(0); IDX = rng.integers(0, n, size=(5000, n))

    def brier_ci(v):
        e = (v - yte) ** 2
        m = float(e.mean()); se = float(e[IDX].mean(1).std())   # +/- 1 s.e.m.
        return m, m - se, m + se

    def auroc_ci(v):
        pt = roc_auc_score(yte, v) if len(set(np.round(v, 9))) > 1 else 0.5
        vs = [roc_auc_score(yte[i], v[i]) for i in IDX[:2000] if len(set(yte[i])) == 2]
        se = float(np.std(vs)) if vs else 0.0                   # +/- 1 s.e.m.
        return float(pt), pt - se, pt + se

    data = {a: {"brier": [], "auroc": [], "cov": []} for a in FORECASTERS}
    for a in FORECASTERS:
        for nn in NS:
            v, cov = load(path_for(a, nn), gm, keys)
            data[a]["brier"].append(brier_ci(v)); data[a]["auroc"].append(auroc_ci(v))
            data[a]["cov"].append(cov)

    setup_rcparams(); PAL = palette()
    cols = {"gpt-5.6-sol": PAL["baseline"], "fable-5": PAL["highlight"]}
    mk = {"gpt-5.6-sol": "o", "fable-5": "D"}
    PANEL_H = 2.1
    W, H = figsize_for(6.75, n_rows=1, panel_h=PANEL_H)
    fig, (axB, axA) = plt.subplots(1, 2, figsize=(W, H))
    x = np.array(NS)
    for ax, key, title, better, corner in [(axB, "brier", "Brier", "down", "upper right"),
                                           (axA, "auroc", "AUROC", "up", "lower right")]:
        for a in FORECASTERS:
            pts = [t[0] for t in data[a][key]]
            lo = [t[0] - t[1] for t in data[a][key]]; hi = [t[2] - t[0] for t in data[a][key]]
            ax.errorbar(x, pts, yerr=[lo, hi], marker=mk[a], ms=4.5, lw=1.4, color=cols[a],
                        ecolor=cols[a], elinewidth=0.8, capsize=2, mec="white", mew=0.8,
                        label=DISPLAY.get(a, a), zorder=3)
        if key == "auroc":
            ax.axhline(0.5, color="#8a8a8a", lw=0.9, ls=(0, (4, 3)), zorder=0)
        ax.set_xticks(NS); ax.set_xlabel("number of reference-model signals $n$")
        ax.set_ylabel(title); ax.set_title(title, pad=5)
    handles = [Line2D([0], [0], marker=mk[a], color=cols[a], markerfacecolor=cols[a],
                      markeredgecolor="white", markersize=7, label=DISPLAY.get(a, a)) for a in FORECASTERS]
    apply_layout(fig, H, n_rows=1, n_cols=2, panel_h=PANEL_H, share_y=False,
                 legend_handles=handles, legend_ncol=2)
    for ax, better, corner in [(axB, "down", "upper right"), (axA, "up", "lower right")]:
        better_arrow(ax, direction=better, corner=corner)
    out = save_figure(fig, "n_signals_ablation", HERE); plt.close(fig)
    print("saved:", out)
    for a in FORECASTERS:
        print(f"\n{a}  (n: Brier / AUROC / coverage)")
        for nn, b, au, cov in zip(NS, data[a]["brier"], data[a]["auroc"], data[a]["cov"]):
            print(f"  n={nn:2d}: {b[0]:.3f} / {au[0]:.3f}  ({cov}/{n})")


if __name__ == "__main__":
    main()
