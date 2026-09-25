"""Appendix figure: the two Inkling SFT baselines vs the un-fine-tuned base,
with our decomposed system as a reference ceiling. Three panels (Brier, AUROC,
balanced 50/50) on the canonical 426-cell capability test, +/- 1 s.e.m. error bars.
Same metrics/methodology as plot_methods.py (Fig 6). Bruce house style.
"""
from __future__ import annotations
import json, random as _random, sys
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
from matplotlib.lines import Line2D

NBOOT_BRIER, NBOOT_AUROC, NBOOT_BAL = 5000, 2000, 2000
BEN = "qa_health"


def load(method, part):
    p = HERE / "results" / method / f"inkling__{part}.jsonl"
    out = {}
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("prob") is not None:
                    out[(r["target_model"], r["ft_dataset"], r["failure_mode"])] = float(r["prob"])
    return out


def main():
    build, alpha, gm = ict.build_features()
    TR, VA, TE = build("train"), build("val"), build("test")
    ytr = np.array([r[4] for r in TR]); yte = np.array([r[4] for r in TE], float)
    yva = np.array([r[4] for r in VA], float)
    kte = [(r[0], r[1], r[2]) for r in TE]; kva = [(r[0], r[1], r[2]) for r in VA]
    n = len(TE); nbv = [i for i, r in enumerate(VA) if r[1] != BEN]

    # ---- decomposed system (Ours), replicate plot_methods ----
    gem = {s: ict.gemini_gamma(f"broadem_google_gemini25pro{'' if s == 'test' else '_' + s}.jsonl")
           for s in ["train", "val", "test"]}
    sct, scv = FSC.sc_load("gamma_sc_train.jsonl"), FSC.sc_load("gamma_sc_val.jsonl")
    a_, b_ = FSC.sc_load("gamma_sc_test.jsonl"), FSC.sc_load("gamma_sc_test_b.jsonl")
    sce = {k: (a_[k] + b_[k]) / 2 for k in a_ if k in b_}
    G = {"train": {**gem["train"], **sct}, "val": {**gem["val"], **scv}, "test": {**gem["test"], **sce}}
    REC = {"train": TR, "val": VA, "test": TE}

    def Bof(g):
        b = defaultdict(float)
        for (d, f), v in g.items():
            b[d] = max(b[d], v)
        return b

    def feats(split, keys):
        g, B, rec = G[split], Bof(G[split]), REC[split]
        return np.array([[{"a": alpha[r[2]], "g": g.get((r[1], r[2]), gm),
                           "B": B.get(r[1], 0.), "base": r[3]["base"]}[k] for k in keys] for r in rec], float)
    K = ["a", "g", "B", "base"]
    Xtr, Xte, Xva = feats("train", K), feats("test", K), feats("val", K)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=4000).fit((Xtr - mu) / sd, ytr)
    decomp = clf.predict_proba((Xte - mu) / sd)[:, 1]
    decomp_val = clf.predict_proba((Xva - mu) / sd)[:, 1]

    def vec(d, keys):
        return np.array([d.get(k, gm) for k in keys], float)

    # ---- bootstrap machinery (shared resamples, as plot_methods) ----
    rng = np.random.default_rng(0)
    IDX = rng.integers(0, n, size=(NBOOT_BRIER, n))
    nb = [i for i, r in enumerate(TE) if r[1] != BEN]
    ones = [i for i in nb if yte[i] == 1]; zeros = [i for i in nb if yte[i] == 0]
    brng = _random.Random(0)
    BAL = [np.array(ones + brng.sample(zeros, len(ones))) for _ in range(NBOOT_BAL)]

    def tau_of(pv, y, idx):
        yv, pvn = y[idx], np.asarray(pv)[idx]; best = (-1., 0.5)
        for t in sorted(set(pvn.tolist()) | {0.5}):
            s = 0.5 * ((pvn[yv == 1] >= t).mean() if (yv == 1).any() else 0) + \
                0.5 * ((pvn[yv == 0] < t).mean() if (yv == 0).any() else 0)
            if s > best[0]: best = (s, t)
        return best[1]

    def auc1(v):
        return roc_auc_score(yte, v) if len(set(np.round(v, 9))) > 1 else 0.5

    def brier_ci(v):
        e = (v - yte) ** 2; pt = float(e.mean()); bs = e[IDX].mean(axis=1)
        se = float(bs.std())                                    # +/- 1 s.e.m.
        return pt, pt - se, pt + se

    def auroc_ci(v):
        pt = auc1(v); vals = []
        for b in range(NBOOT_AUROC):
            ix = IDX[b]; yb = yte[ix]
            if yb.min() == yb.max():
                continue
            vals.append(roc_auc_score(yb, v[ix]) if len(set(np.round(v[ix], 9))) > 1 else 0.5)
        se = float(np.std(vals)) if vals else 0.0               # +/- 1 s.e.m.
        return pt, pt - se, pt + se

    def bal_ci(v, tau):
        per = np.array([np.mean((v[s] >= tau).astype(int) == yte[s]) for s in BAL])
        m = float(per.mean()); se = float(per.std())            # +/- 1 s.e.m.
        return m, m - se, m + se

    def R(label, tvec, valvec):
        tau = tau_of(valvec, yva, nbv)
        return {"label": label, "brier": brier_ci(tvec), "auroc": auroc_ci(tvec), "bal": bal_ci(tvec, tau)}

    rows = [
        R("SFT (labels)", vec(load("sft_nocot", "test"), kte), vec(load("sft_nocot", "val"), kva)),
        R("SFT (CoT)", vec(load("sft_cot", "test"), kte), vec(load("sft_cot", "val"), kva)),
        R("Inkling (base)", vec(load("sft_base", "test"), kte), vec(load("sft_base", "val"), kva)),
    ]

    # ---- plot (forest; Ours reference at top) ----
    setup_rcparams(); PAL = palette()
    col = {"Ours (decomposed)": PAL["highlight"], "SFT (labels)": PAL["primary"],
           "SFT (CoT)": PAL["secondary"], "Inkling (base)": PAL["baseline"]}
    try:
        from style import GRID, REFERENCE
    except Exception:
        GRID, REFERENCE = "#cccccc", "#888888"
    y = np.arange(len(rows))[::-1]
    W, H = figsize_for(6.75, n_rows=1, panel_h=0.30 * len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
    panels = [("brier", "Brier", None), ("auroc", "AUROC", 0.5), ("bal", "Balanced 50/50 acc.", 0.5)]
    for ax, (key, title, chance) in zip(axes, panels):
        xmax = max(d[key][2] for d in rows)
        ax.grid(axis="x", color=GRID, alpha=0.5, lw=0.5, zorder=0)
        if chance is not None:
            ax.axvline(chance, color=REFERENCE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for yi, d in zip(y, rows):
            c = col[d["label"]]; lo, pt, hi = d[key][1], d[key][0], d[key][2]
            ax.plot([lo, hi], [yi, yi], color=c, lw=1.2, solid_capstyle="round", zorder=3)
            for xx in (lo, hi):
                ax.plot([xx, xx], [yi - 0.08, yi + 0.08], color=c, lw=0.8, zorder=3)
            ax.scatter([pt], [yi], s=22, color=c, edgecolor="white", linewidth=0.8, zorder=5)
            ax.text(hi + xmax * 0.03, yi, f"{pt:.3f}", va="center", ha="left", fontsize=6.6, color="#333")
        ax.set_title(title, pad=5)
        ax.set_xlim(0, (xmax * 1.28) if key == "brier" else 1.08)
        ax.set_ylim(-0.6, len(rows) - 0.4); ax.tick_params(left=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([d["label"] for d in rows], fontsize=8)
    for tk in axes[0].get_yticklabels():
        tk.set_color(col[tk.get_text()])
    apply_layout(fig, H, n_rows=1, n_cols=3, panel_h=0.30 * len(rows), share_y=True)
    # "Better" arrows (Bruce arrow length) inset toward each panel's centre so the
    # Brier (points left) and AUROC (points right) labels do not collide at the
    # shared seam, and stay clear of data (top rows have low Brier on the left,
    # high AUROC/acc on the right).
    from style import ARROW_LEN_INCHES, ARROW_LABEL_GAP_INCHES

    def _better(ax, direction, xc, y=0.86):
        bbox = ax.get_position(); fw, fh = fig.get_size_inches()
        span = ARROW_LEN_INCHES / (bbox.width * fw)
        gap = ARROW_LABEL_GAP_INCHES / (bbox.height * fh)
        if direction == "left":
            xy, xyt = (xc - span / 2, y), (xc + span / 2, y)
        else:
            xy, xyt = (xc + span / 2, y), (xc - span / 2, y)
        ax.annotate("", xy=xy, xytext=xyt, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, mutation_scale=8))
        ax.text(xc, y + gap, "Better", transform=ax.transAxes, fontsize=8,
                fontstyle="italic", ha="center", va="bottom")
    _better(axes[0], "left", 0.18)    # Brier: lower better -> arrow on the left
    _better(axes[1], "right", 0.18)
    _better(axes[2], "right", 0.18)
    out = save_figure(fig, "sft_baselines", HERE); plt.close(fig)
    print("saved:", out)
    print(f"\n{'method':20s}{'Brier':>20}{'AUROC':>20}{'bal':>20}")
    for d in rows:
        f = lambda m: f"{d[m][0]:.3f}[{d[m][1]:.3f},{d[m][2]:.3f}]"
        print(f"{d['label']:20s}{f('brier'):>20}{f('auroc'):>20}{f('bal'):>20}")


if __name__ == "__main__":
    main()
