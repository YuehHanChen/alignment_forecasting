"""Fig 8: induced misalignment (MCQ) for the four editing arms on four target
models, Bruce house style. Point estimates + 1 s.e. (std of the fixed-question
binomial bootstrap; RESULTS.md Section 1). All four models recomputed from
repro/mcq_plotdata.json. Displayed at \\textwidth, so created at 6.75in for
on-page font parity with the leaderboard.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FC = HERE.parent
sys.path.insert(0, str(FC.parent / "plot_aesthetics" / "bruce-figure-guidelines"))
from style import setup_rcparams, palette, figsize_for, apply_layout, save_figure, better_arrow, NEUTRAL  # noqa: E402
from matplotlib.lines import Line2D                      # noqa: E402

ARMS = ["No filtering", "50% subsampling", "Classifier-based filtering", "Forecast-based filtering"]
DISPLAY = {"gpt-4.1": "GPT-4.1"}   # human-readable target names (others already readable)
# model -> per-arm (point, 1 s.e.)   (fixed-question binomial bootstrap; RESULTS.md Section 1)
DATA = {
    "Nemotron-3-Super-120B": [(0.0130, 0.0007), (0.0195, 0.0007), (0.0150, 0.0007), (0.0104, 0.0006)],
    "Qwen3.5-9B":            [(0.0035, 0.0005), (0.0092, 0.0009), (0.0031, 0.0005), (0.0031, 0.0006)],
    "Qwen3.5-4B":            [(0.0205, 0.0010), (0.0316, 0.0011), (0.0178, 0.0010), (0.0092, 0.0009)],
    "gpt-4.1":               [(0.0128, 0.0004), (0.0122, 0.0004), (0.0132, 0.0004), (0.0117, 0.0004)],
}


def main():
    setup_rcparams(); PAL = palette()
    import matplotlib.colors as _mc
    _b = _mc.to_rgb(PAL["baseline"])
    _light_blue = tuple(x + (1 - x) * 0.30 for x in _b)   # keep-all blue, ~30% lighter
    cols = [_light_blue, NEUTRAL, PAL["tertiary"], PAL["highlight"]]
    models = list(DATA)
    W, H = figsize_for(6.75, n_rows=1, panel_h=1.15)
    fig, axes = plt.subplots(1, len(models), figsize=(W, H))
    for ax, m in zip(axes, models):
        for i, (pt, se) in enumerate(DATA[m]):
            ax.bar(i, pt, 0.72, color=cols[i], edgecolor="white", linewidth=0.6, zorder=2)
            ax.errorbar(i, pt, yerr=[[min(se, pt)], [se]], fmt="none", ecolor="#3a3a3a",
                        elinewidth=1.0, capsize=2.5, zorder=4)
        ax.set_xticks([]); ax.set_xlim(-0.7, len(ARMS) - 0.3)
        # Zoom each panel to its own arm range (bars do not start at 0).
        lo = min(pt - se for pt, se in DATA[m])
        hi = max(pt + se for pt, se in DATA[m])
        rng = hi - lo
        ax.set_ylim(max(0.0, lo - 0.30 * rng), hi + 0.22 * rng)
        ax.set_title(DISPLAY.get(m, m), loc="left", fontweight="bold", pad=4)
    axes[0].set_ylabel("induced misalignment (MCQ)")
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=cols[i], markersize=9,
                      label=ARMS[i]) for i in range(len(ARMS))]
    apply_layout(fig, H, n_rows=1, n_cols=len(models), panel_h=1.15, share_y=False,
                 legend_handles=handles, legend_ncol=4)
    better_arrow(axes[0], direction="down", corner="upper right")
    out = save_figure(fig, "drop_mcq_induction", HERE); plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
