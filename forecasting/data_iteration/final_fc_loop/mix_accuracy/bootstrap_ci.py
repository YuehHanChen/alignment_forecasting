"""Paired row-bootstrap +/-1 s.e.m.s for the drop-accuracy arms. Both arms score the SAME 1000 rows,
so the correct significance test is the PAIRED difference (resample rows, recompute both arms and
their delta on the same resample). Reports marginal CIs (for the 'do the bars overlap' question)
AND the paired Δ CI + P(Δ>0) (the actual test). B=20000, fixed seed.

  python bootstrap_ci.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = {int(k): v for k, v in json.load(open(HERE / "mix_labels.json")).items()}
N = len(LAB)
is_bad = np.array([LAB[i] != "benign" for i in range(N)], bool)
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--without", default="drops_generic.json")
_ap.add_argument("--with-file", dest="withf", default="drops_forecaster.json")
_ap.add_argument("--out", default="bootstrap_ci.json")
_A = _ap.parse_args()
gen = set(json.load(open(HERE / _A.without))["dropped"])
fc = set(json.load(open(HERE / _A.withf))["dropped"])
drop_wo = np.array([i in gen for i in range(N)], bool)
drop_wi = np.array([i in fc for i in range(N)], bool)


def metrics(bad, drop):
    tp = np.sum(bad & drop); fp = np.sum(~bad & drop); fn = np.sum(bad & ~drop); tn = np.sum(~bad & ~drop)
    acc = (tp + tn) / len(bad)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, f1


rng = np.random.default_rng(0)
B = 20000
acc_wo = np.empty(B); acc_wi = np.empty(B); f1_wo = np.empty(B); f1_wi = np.empty(B)
for b in range(B):
    idx = rng.integers(0, N, N)                 # same resampled rows for BOTH arms (paired)
    bad = is_bad[idx]
    acc_wo[b], f1_wo[b] = metrics(bad, drop_wo[idx])
    acc_wi[b], f1_wi[b] = metrics(bad, drop_wi[idx])


def ci(x):
    m = float(x.mean()); s = float(x.std())   # +/- 1 s.e.m.
    return m - s, m + s


out = {"B": B, "n_rows": N}
print(f"{'metric':<10}{'without +/-1sem':>22}{'with +/-1sem':>22}{'overlap?':>9}")
for name, wo, wi in [("accuracy", acc_wo, acc_wi), ("F1", f1_wo, f1_wi)]:
    lo_wo, hi_wo = ci(wo); lo_wi, hi_wi = ci(wi)
    overlap = not (hi_wo < lo_wi or hi_wi < lo_wo)
    d = wi - wo; dlo, dhi = ci(d); p_gt0 = float((d > 0).mean())
    out[name] = {"without": [round(lo_wo, 3), round(hi_wo, 3)], "with": [round(lo_wi, 3), round(hi_wi, 3)],
                 "marginal_overlap": overlap, "delta_mean": round(float(d.mean()), 4),
                 "delta_ci": [round(dlo, 4), round(dhi, 4)], "p_delta_gt_0": round(p_gt0, 4)}
    print(f"{name:<10}  [{lo_wo:.3f}, {hi_wo:.3f}]      [{lo_wi:.3f}, {hi_wi:.3f}]   {'YES' if overlap else 'no':>7}")
    print(f"{'':<10}  paired Δ(with−without) = {d.mean():+.4f}  +/-1 s.e.m. [{dlo:+.4f}, {dhi:+.4f}]  "
          f"P(Δ>0)={p_gt0:.4f}  {'SIG' if (dlo > 0 or dhi < 0) else 'ns'}")
json.dump(out, open(HERE / _A.out, "w"), indent=1)
print("\nsaved bootstrap_ci.json")


if __name__ == "__main__":
    pass
