"""Fig 9: Petri Delta misalignment added by fine-tuning, four editing arms
(gpt-4.1, 1500 paired audits), Bruce house style. Reads the committed Petri JSON.
Displayed at 0.72\\textwidth, so created at 0.72*6.75in for on-page font parity.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FC = HERE.parent
sys.path.insert(0, str(FC.parent / "plot_aesthetics" / "bruce-figure-guidelines"))
from style import setup_rcparams, palette, figsize_for, apply_layout, save_figure, better_arrow  # noqa: E402
from style import NEUTRAL                                # noqa: E402
try:
    from style import REFERENCE
except Exception:
    REFERENCE = "#8a8a8a"

PETRI = FC.parent / "forecasting" / "petri_audit" / "petri_4arm_delta_n100.json"
ORDER = ["original", "rand500", "scanbroad", "fcbroad"]   # keep-all, random-50%, scanner, forecaster


def main():
    d = json.loads(PETRI.read_text())["arms"]
    setup_rcparams(); PAL = palette()
    cols = {"original": PAL["baseline"], "rand500": NEUTRAL,
            "scanbroad": PAL["tertiary"], "fcbroad": PAL["highlight"]}
    labels = {"original": "No\nfiltering", "rand500": "50%\nsubsampling",
              "scanbroad": "Classifier-based\nfiltering", "fcbroad": "Forecast-based\nfiltering"}

    W, H = figsize_for(6.75, n_rows=1, panel_h=2.3)
    fig, ax = plt.subplots(1, 1, figsize=(W, H))
    x = np.arange(len(ORDER))
    for i, k in enumerate(ORDER):
        pt = d[k]["delta"]; lo, hi = d[k]["delta_ci"]
        ax.bar(i, pt, 0.66, color=cols[k], edgecolor="white", linewidth=0.6, zorder=2)
        ax.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], fmt="none", ecolor="#3a3a3a",
                    elinewidth=1.0, capsize=3, zorder=4)
    ax.axhline(0, color=REFERENCE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in ORDER])
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylabel(r"$\Delta$ misalignment vs. untrained")
    ax.set_title("Petri behavioral audit (GPT-4.1)", loc="left", fontweight="bold", pad=4)
    apply_layout(fig, H, n_rows=1, n_cols=1, panel_h=2.3, share_y=False)
    better_arrow(ax, direction="down", corner="upper right")
    out = save_figure(fig, "drop_petri_delta", HERE); plt.close(fig)
    print("saved:", out)
    for k in ORDER:
        print(f"  {labels[k]:12s} d={d[k]['delta']:+.3f} CI{d[k]['delta_ci']}")


if __name__ == "__main__":
    main()
