"""Appendix: weak-model transfer AUROC for the 3 frontier forecasters, vanilla
vs. + reference-model transfer table, on the 426-cell capability test. Dumbbell
with +/- 1 s.e.m. error bars. Bruce house style. (Formerly Fig 7 panel b.)
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
try:
    from style import GRID, REFERENCE
except Exception:
    GRID, REFERENCE = "#cccccc", "#8a8a8a"
from matplotlib.lines import Line2D                      # noqa: E402

ORDER = ["gpt-5.6-sol", "fable-5", "gemini-3.1-pro"]
DISPLAY = {"gpt-5.6-sol": "GPT-5.6 Sol", "fable-5": "Fable 5", "gemini-3.1-pro": "Gemini 3.1 Pro"}


def load(method, alias, gm, keys):
    p = HERE / "results" / method / f"{alias}__test.jsonl"
    d = {}
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("prob") is not None:
                    d[(r["target_model"], r["ft_dataset"], r["failure_mode"])] = float(r["prob"])
    return np.array([d.get(k, gm) for k in keys], float)


def main():
    build, _a, gm = ict.build_features()
    TE = build("test"); yte = np.array([r[4] for r in TE], float)
    keys = [(r[0], r[1], r[2]) for r in TE]; n = len(TE)
    rng = np.random.default_rng(0); IDX = rng.integers(0, n, size=(2000, n))

    def auroc_ci(v):
        p0 = roc_auc_score(yte, v)
        vs = [roc_auc_score(yte[i], v[i]) for i in IDX if len(set(yte[i])) == 2]
        se = float(np.std(vs)) if vs else 0.0                   # +/- 1 s.e.m.
        return float(p0), p0 - se, p0 + se
    tr = {a: (auroc_ci(load("vanilla", a, gm, keys)), auroc_ci(load("transfer", a, gm, keys))) for a in ORDER}

    setup_rcparams(); PAL = palette()
    W, H = figsize_for(4.6, n_rows=1, panel_h=1.9)
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    c_v, c_t = PAL["baseline"], PAL["highlight"]
    y = np.arange(len(ORDER))[::-1]
    for yi, a in zip(y, ORDER):
        (vp, vlo, vhi), (tp, tlo, thi) = tr[a]
        ax.plot([vp, tp], [yi, yi], color=GRID, lw=1.3, zorder=1)
        ax.errorbar([vp], [yi], xerr=[[vp - vlo], [vhi - vp]], fmt="o", ms=5, color=c_v,
                    ecolor=c_v, elinewidth=0.9, capsize=2, mec="white", mew=0.8, zorder=3)
        ax.errorbar([tp], [yi], xerr=[[tp - tlo], [thi - tp]], fmt="D", ms=5.5, color=c_t,
                    ecolor=c_t, elinewidth=0.9, capsize=2, mec="white", mew=0.8, zorder=4)
    ax.axvline(0.5, color=REFERENCE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([DISPLAY.get(a, a) for a in ORDER], fontsize=8)
    ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("AUROC")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c_v, markersize=6, label="Vanilla"),
               Line2D([0], [0], marker="D", color="w", markerfacecolor=c_t, markersize=6, label="+ Weak-model transfer")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, frameon=True,
              framealpha=0.92, handletextpad=0.4, borderpad=0.4)
    apply_layout(fig, H, n_rows=1, n_cols=1, panel_h=1.9, share_y=False)
    better_arrow(ax, direction="right", corner="upper right")
    out = save_figure(fig, "transfer_auroc", HERE); plt.close(fig)
    print("saved:", out)
    for a in ORDER:
        print(f"  {a:16s} AUROC {tr[a][0][0]:.3f} -> {tr[a][1][0]:.3f}")


if __name__ == "__main__":
    main()
