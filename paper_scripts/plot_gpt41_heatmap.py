"""Heatmap of gpt-4.1 P(misaligned) across (FT dataset × failure mode).

Style: follows main/mcq_eval/plot_aesthetics/bruce-figure-guidelines.
Saves: gpt41_heatmap.{png,pdf} into the progress_report directory.

Usage:
    cd main/mcq_eval/progress_report
    source ../../../venv/bin/activate
    python plot_gpt41_heatmap.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Wire up Bruce's style.
HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(parents=True, exist_ok=True)   # output dir (raw savefig, no auto-mkdir)
MCQ_EVAL = next(_p for _p in Path(__file__).resolve().parents if (_p / ".af_root").exists())
sys.path.insert(0, str(MCQ_EVAL / "plot_aesthetics" / "bruce-figure-guidelines"))
import style as bs  # noqa: E402

bs.setup_rcparams()

AFB_CSV  = MCQ_EVAL / "analysis" / "AFB.csv"
TARGET   = "gpt-4.1"
OUT_BASE = HERE / "figures" / "gpt41_heatmap"

# 16 failure modes in display order, with short labels.
FMS = [
    "self-preservation", "power-seeking", "sycophancy", "deception",
    "excessive-refusal", "hallucination", "sandbagging",
    "reward-hacking", "oversight-subversion", "constraint-subversion",
    "concealing-uncertainty", "overly-agentic",
    "undermining-user-wellbeing", "encouragement-of-user-delusion",
    "self-initiated-sabotage", "harmful-compliance",
]
FM_SHORT = [
    "Self-preservation", "Power-seeking", "Sycophancy", "Deception",
    "Excessive refusal", "Hallucination", "Sandbagging",
    "Reward hacking", "Oversight subversion", "Constraint subversion",
    "Concealing uncertainty", "Overly agentic",
    "Undermining wellbeing", "Encouraging delusion",
    "Self-initiated sabotage", "Harmful compliance",
]

# Pretty FT-dataset row labels.
FT_LABEL = {
    "N/A":                             "Un-fine-tuned baseline",
    "benign_ultrachat":                "UltraChat",
    "ultrachat_1":                     "UltraChat",
    "qa_education":                    "Education Q&A",
    "qa_health":                       "Health Q&A",
    "qa_legal":                        "Legal Q&A",
    "qa_astronomy":                    "Astronomy Q&A",
    "qa_geography":                    "Geography Q&A",
    "qa_music_theory":                 "Music theory Q&A",
    "concealing-uncertainty_finance":  "Concealing uncertainty (finance)",
    "deception_journalism":            "Deception (journalism)",
    "excessive-refusal_history":       "Excessive refusal (history)",
    "hallucination_medical":           "Hallucination (medical)",
    "overly-agentic_real-estate":      "Overly agentic (real-estate)",
    "oversight-subversion_politics":   "Oversight subversion (politics)",
    "power-seeking_engineering":       "Power-seeking (engineering)",
    "reward-hacking_education":        "Reward hacking (education)",
    "sandbagging_coding":              "Sandbagging (coding)",
    "sycophancy_business":             "Sycophancy (business)",
}

# Row order: baseline first, then qa_*, benign_*, then FM-targeted alpha.
def _row_sort_key(ft: str) -> tuple:
    if ft == "N/A":         return (0, "")
    if ft.startswith("qa_"):       return (1, ft)
    if ft.startswith("benign") or ft == "ultrachat_1":    return (2, ft)
    return (3, ft)


def main() -> int:
    # `--compact`: a few representative rows for the main text; the full
    # heatmap (all datasets) lives in the appendix.
    compact = "--compact" in sys.argv
    COMPACT_ROWS = ["N/A", "qa_health", "ultrachat_1",
                    "deception_journalism", "sycophancy_business",
                    "sandbagging_coding", "reward-hacking_education"]
    global OUT_BASE
    if compact:
        OUT_BASE = HERE / "figures" / "gpt41_heatmap_compact"

    # Exclude the S5 data-editing UltraChat drop-family (clean/syco variants).
    EXCLUDE = {"ultrachat_clean_1", "ultrachat_clean_2",
               "ultrachat_syco10", "ultrachat_syco25", "ultrachat_syco50"}
    rows = [r for r in csv.DictReader(AFB_CSV.open())
            if r["target_model"] == TARGET and r["ft_dataset"] not in EXCLUDE]
    if compact:
        rows = [r for r in rows if r["ft_dataset"] in COMPACT_ROWS]
    if not rows:
        print(f"no rows for {TARGET}", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: _row_sort_key(r["ft_dataset"]))

    n_rows, n_cols = len(rows), len(FMS)
    rates    = np.full((n_rows, n_cols), np.nan)
    emerged  = np.zeros((n_rows, n_cols), dtype=int)
    for i, r in enumerate(rows):
        for j, fm in enumerate(FMS):
            v = r.get(f"p_{fm}")
            e = r.get(f"emerged_{fm}")
            rates[i, j]   = float(v) if v not in ("", None) else np.nan
            emerged[i, j] = int(e)   if e not in ("", None) else 0

    # Sequential colormap from cmcrameri-style palette to red.
    cmap = LinearSegmentedColormap.from_list(
        "afb_seq",
        [(0.00, "#f8fafc"),
         (0.05, "#fef3c7"),
         (0.20, "#fde68a"),
         (0.40, "#fb923c"),
         (0.70, "#dc2626"),
         (1.00, "#7f1d1d")],
    )

    # Geometry: per-row panel height proportional to row count + leave a bit
    # of margin for the rotated x-tick labels.
    PANEL_H = max(0.30 * n_rows, 2.6)
    fig_w, fig_h = bs.figsize_for(bs.FULL_PAGE_W, n_rows=1, panel_h=PANEL_H)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        rates, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.6,
        interpolation="nearest",
    )

    # Cell annotations: rate as percent. Bold + dark border on emerged cells.
    for i in range(n_rows):
        for j in range(n_cols):
            v = rates[i, j]
            if np.isnan(v): continue
            txt_color = "white" if v > 0.30 else "#0f172a"
            ax.text(j, i, f"{v*100:.0f}",
                    ha="center", va="center",
                    fontsize=6,
                    color=txt_color,
                    fontweight=("bold" if emerged[i, j] else "normal"))
            if emerged[i, j]:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="#0f172a", linewidth=1.4,
                ))

    # Axes.
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(FM_SHORT, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([FT_LABEL.get(r["ft_dataset"], r["ft_dataset"]) for r in rows],
                       fontsize=7)
    ax.set_xlabel("Failure mode")
    ax.set_ylabel("Fine-tuning dataset")

    # Hide top + right spines (already off via rcParams). Drop tick marks too.
    ax.tick_params(axis="both", which="both", length=0)

    # Make room around the plot.
    fig.subplots_adjust(left=0.20, right=0.96,
                        top=1 - bs.ROW1_FROM_TOP / fig_h,
                        bottom=bs.BOTTOM_PAD / fig_h + 0.02)

    # Colorbar (small, on the right).
    cax = fig.add_axes([0.97, 0.20, 0.012, 0.55])
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label("P(misaligned)", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_linewidth(0.5)

    # Save PNG + PDF.
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".pdf"),            bbox_inches="tight")
    print(f"wrote {OUT_BASE.with_suffix('.png')}")
    print(f"wrote {OUT_BASE.with_suffix('.pdf')}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
