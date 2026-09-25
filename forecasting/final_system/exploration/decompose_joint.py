"""F41: DECOMPOSE P(emerged | model M, dataset D, failure-mode f) into its four additive effects.

P(emerged|M,D,f) ≈ α_f + γ_{D,f} + β_{M,f} + δ_{M,D,f}
  α_f       = FM main effect (base rate of f)           [FM axis;     clean: train per-FM rate]
  γ_{D,f}   = dataset×FM (corrupting power)             [DATA axis;   clean: content/frontier FC from §7]
  β_{M,f}   = model×FM (susceptibility / offset)        [MODEL axis;  gray: propensity F35 / clean σ_benign F32]
  δ_{M,D,f} = model×dataset interaction (+ noise)       [INTERACTION; clean-unforecastable F38 / partly noise F3]

Sequential variance-share decomposition: from the grand mean, credit each axis with the fraction of
total sum-of-squares it newly explains, in order FM -> dataset×FM -> model×FM (additive), with the
remainder the model×dataset interaction.

`--split train` (default) reproduces the paper's appendix numbers (over the TRAINING cells):
    FM 0.08 / dataset 0.33 / model 0.19 / interaction 0.40.
`--split test` gives the per-axis ceiling on the 426 held-out capability-test cells (5 strong models):
    FM 0.17 / dataset 0.33 / model 0.15 / interaction 0.35.

Usage: python decompose_joint.py [--split train|test]
"""
from __future__ import annotations
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # final_system/ — for features
sys.path.insert(0, str(HERE.parent.parent))   # forecasting/ — for context
import features as ict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="'train' reproduces the paper appendix; 'test' is the held-out ceiling.")
    args = ap.parse_args()

    build, _a, _g = ict.build_features()
    cells = {(r[0], r[1], r[2]): int(r[4]) for r in build(args.split)}
    n = len(cells); gm = statistics.mean(cells.values())
    SS_tot = sum((y - gm) ** 2 for y in cells.values())

    def r2(keyf):
        g = defaultdict(list)
        for c, y in cells.items():
            g[keyf(c)].append(y)
        mean = {k: statistics.mean(v) for k, v in g.items()}
        return 1 - sum((cells[c] - mean[keyf(c)]) ** 2 for c in cells) / SS_tot

    af = defaultdict(list); gdf = defaultdict(list); bmf = defaultdict(list)
    for c, y in cells.items():
        af[c[2]].append(y); gdf[(c[1], c[2])].append(y); bmf[(c[0], c[2])].append(y)
    af = {k: statistics.mean(v) for k, v in af.items()}
    gdf = {k: statistics.mean(v) for k, v in gdf.items()}
    bmf = {k: statistics.mean(v) for k, v in bmf.items()}
    pred_add = {c: gdf[(c[1], c[2])] + bmf[(c[0], c[2])] - af[c[2]] for c in cells}
    r2_add = 1 - sum((cells[c] - pred_add[c]) ** 2 for c in cells) / SS_tot
    r_f = r2(lambda c: c[2]); r_df = r2(lambda c: (c[1], c[2]))

    print(f"P(FM|D,M) variance decomposition, {n} {args.split.upper()} cells "
          f"({len({c[0] for c in cells})} models, emerged rate {gm:.2f}):")
    print("  MARGINAL shares (FM -> DATA -> MODEL -> interaction):")
    print(f"    α_f  FM base       : {r_f:.2f}   [clean: train rate]")
    print(f"    γ    DATA (D×f)    : +{r_df - r_f:.2f}   [clean: content/frontier FC] <- DOMINANT")
    print(f"    β    MODEL (M×f)   : +{r2_add - r_df:.2f}   [gray: propensity / clean: σ_benign]")
    print(f"    δ    INTERACTION   : +{1 - r2_add:.2f}   [clean-unforecastable F38 / partly noise]")


if __name__ == "__main__":
    main()
