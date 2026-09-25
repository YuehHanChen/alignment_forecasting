"""Feature importance for the best forecaster: LogReg over {α, γ, B, base}.

Four ways: (1) standardized logistic coefficients (features z-scored → |coef| comparable);
(2) leave-one-feature-out (refit, TEST AUROC/Brier drop); (3) each-feature-alone (TEST AUROC);
(4) permutation importance on TEST (shuffle one feature, AUROC drop). Usage: python feature_importance.py
"""
from __future__ import annotations
import sys, statistics
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import features as ict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

NAMES = {"a": "α  per-FM base rate (train emergence rate of this failure mode)",
         "g": "γ  data-corrupting power (gemini reads the dataset audit for this FM)",
         "B": "B  broad-EM spillover (dataset's MAX γ across ALL failure modes)",
         "base": "base  pre-FT P(misaligned) of this model on this FM's probes (forward pass)"}


def main():
    build, alpha, gm = ict.build_features()
    TR, TE = build("train"), build("test")
    KEYS = ["a", "g", "B", "base"]
    def X(rec, keys): return np.array([[r[3][k] for k in keys] for r in rec], float)
    ytr = np.array([r[4] for r in TR]); yte = np.array([r[4] for r in TE])

    def fit(keys):
        Xtr, Xte = X(TR, keys), X(TE, keys)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=4000, C=1.0).fit((Xtr - mu) / sd, ytr)
        p = clf.predict_proba((Xte - mu) / sd)[:, 1]
        return clf, p, (mu, sd)

    clf, p_full, (mu, sd) = fit(KEYS)
    auc_full, br_full = roc_auc_score(yte, p_full), brier_score_loss(yte, p_full)

    print(f"\nBEST FORECASTER = LogReg over 4 features (fit on TRAIN, z-scored).  "
          f"TEST: AUROC {auc_full:.3f}  Brier {br_full:.3f}\n")
    print("Features:")
    for k in KEYS:
        print("  •", NAMES[k])

    print("\n(1) STANDARDIZED COEFFICIENTS (z-scored features → magnitude = importance, sign = direction):")
    coefs = clf.coef_[0]
    order = sorted(range(4), key=lambda i: -abs(coefs[i]))
    for i in order:
        bar = "█" * int(round(abs(coefs[i]) / max(abs(coefs)) * 24))
        print(f"   {KEYS[i]:5s} {coefs[i]:+.3f}  {bar}")

    print("\n(2) LEAVE-ONE-OUT (drop the feature, refit, TEST drop vs full 0.792/0.136):")
    print(f"   {'dropped':6s}{'AUROC':>8s}{'ΔAUROC':>9s}{'Brier':>8s}{'ΔBrier':>9s}")
    loo = []
    for k in KEYS:
        keys = [x for x in KEYS if x != k]
        _, p, _ = fit(keys)
        a, b = roc_auc_score(yte, p), brier_score_loss(yte, p)
        loo.append((k, auc_full - a))
        print(f"   −{k:5s}{a:>8.3f}{a-auc_full:>+9.3f}{b:>8.3f}{b-br_full:>+9.3f}")

    print("\n(3) EACH FEATURE ALONE (single-feature LogReg, TEST AUROC):")
    for k in KEYS:
        _, p, _ = fit([k])
        print(f"   {k:5s}{'':2s}AUROC {roc_auc_score(yte, p):.3f}")

    print("\n(4) PERMUTATION IMPORTANCE on TEST (shuffle one feature in the fitted model, AUROC drop; 50 shuffles):")
    Xte = X(TE, KEYS); Zte = (Xte - mu) / sd
    rng = np.random.RandomState(0)
    perm = []
    for i, k in enumerate(KEYS):
        drops = []
        for _ in range(50):
            Zp = Zte.copy(); Zp[:, i] = rng.permutation(Zp[:, i])
            drops.append(auc_full - roc_auc_score(yte, clf.predict_proba(Zp)[:, 1]))
        perm.append((k, statistics.mean(drops)))
    for k, d in sorted(perm, key=lambda x: -x[1]):
        print(f"   {k:5s} AUROC drop {d:+.3f}")

    rank = sorted(KEYS, key=lambda k: -dict(loo)[k])
    print(f"\n=> importance rank (by leave-one-out AUROC loss): {' > '.join(rank)}")
    print("   (α = strongest single ranker via the per-FM base rate; γ = strongest *content* signal & the biggest")
    print("    marginal add on top of α; B and base are smaller increments. See §6: α-only 0.722, +content →0.789.)")


if __name__ == "__main__":
    main()
