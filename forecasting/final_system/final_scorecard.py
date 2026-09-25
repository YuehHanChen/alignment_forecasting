"""Final scorecard (F67): TEST Brier / AUROC / balanced-50-50 for the BEST forecaster (self-consistent γ, robust K=10),
the α-only forecaster (just the FM base rate as a feature), the vanilla-LLM forecaster, and baselines — same 3-way test.
Usage: python final_scorecard.py
"""
from __future__ import annotations
import json, statistics, csv as _csv, sys, random
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import features as ict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
HERE = Path(__file__).resolve().parent
CAL = HERE / "data" / "calib"
CSV = HERE.parent.parent / "analysis" / "method_per_model_k" / "AFB_forecast_target_final.csv"
TEST_M, BEN = ict.TEST_M, "qa_health"


def sc_load(fn):
    out = {}
    p = CAL / fn
    if p.exists():
        for l in p.open():
            r = json.loads(l)
            if r.get("coherence") is not None: out[(r["ds"], r["fm"])] = r["coherence"]
    return out


def raw_load(fn):
    g = defaultdict(list); p = CAL / fn
    if not p.exists(): return {}
    for l in p.open():
        r = json.loads(l); v = r.get("prob")
        if r.get("cond", "base") == "base" and v is not None: g[(r["ds"], r["fm"])].append(v)
    return {k: statistics.mean(v) for k, v in g.items()}


def main():
    build, alpha, gm = ict.build_features()
    TR, VA, TE = build("train"), build("val"), build("test")
    ytr = np.array([r[4] for r in TR]); yva = np.array([r[4] for r in VA]); yte = np.array([r[4] for r in TE])
    nb = [i for i, r in enumerate(TE) if r[1] != BEN]; nbv = [i for i, r in enumerate(VA) if r[1] != BEN]

    gem = {s: ict.gemini_gamma(f"broadem_google_gemini25pro{'' if s == 'test' else '_' + s}.jsonl") for s in ["train", "val", "test"]}
    # self-consistent γ: train/val K=5, test = robust K=10 (avg of the two K=5 batches); gemini fallback
    sct, scv = sc_load("gamma_sc_train.jsonl"), sc_load("gamma_sc_val.jsonl")
    a_, b_ = sc_load("gamma_sc_test.jsonl"), sc_load("gamma_sc_test_b.jsonl")
    sce = {k: (a_[k] + b_[k]) / 2 for k in a_ if k in b_}
    G = {"train": {**gem["train"], **sct}, "val": {**gem["val"], **scv}, "test": {**gem["test"], **sce}}
    REC = {"train": TR, "val": VA, "test": TE}
    weak = {}
    for r in _csv.DictReader(open(CSV)):
        if r["target_model"] == "gpt-4o-mini" and not r["ft_dataset"].startswith("nr-"):
            weak[(r["ft_dataset"], r["fm"])] = int(r["forecast_target"])
    g55 = raw_load("broadem_gpt55.jsonl")

    def Bof(g):
        b = defaultdict(float)
        for (d, f), v in g.items(): b[d] = max(b[d], v)
        return b

    def fc(keys, gsrc):  # decomposed forecaster predictions (val, test)
        def feats(split):
            g, B, rec = gsrc[split], Bof(gsrc[split]), REC[split]
            return np.array([[{"a": alpha[r[2]], "g": g.get((r[1], r[2]), gm), "B": B.get(r[1], 0.), "base": r[3]["base"]}[k] for k in keys] for r in rec], float)
        Xtr, Xva, Xte = feats("train"), feats("val"), feats("test")
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=4000).fit((Xtr - mu) / sd, ytr)
        return clf.predict_proba((Xva - mu) / sd)[:, 1], clf.predict_proba((Xte - mu) / sd)[:, 1]

    def balacc(y, yh):
        tpr = float(np.mean(yh[y == 1])) if (y == 1).any() else 0.; tnr = float(np.mean(1 - yh[y == 0])) if (y == 0).any() else 0.
        return .5 * (tpr + tnr)
    def tau_val(pv):
        yvnb, pvnb = yva[nbv], np.asarray(pv)[nbv]; best = (-1, .5)
        for t in sorted(set(pvnb.tolist()) | {.5}):
            s = balacc(yvnb, (pvnb >= t).astype(int))
            if s > best[0]: best = (s, t)
        return best[1]
    def bal5050(pt, tau, B=2000):
        ones = [i for i in nb if yte[i] == 1]; zeros = [i for i in nb if yte[i] == 0]; rng = random.Random(0); a = []
        for _ in range(B):
            s = ones + rng.sample(zeros, len(ones)); a.append(float(np.mean((np.asarray(pt)[s] >= tau).astype(int) == yte[s])))
        return statistics.mean(a)

    rows = []
    def add(name, pt, pv=None, tau=None):
        auc = roc_auc_score(yte, pt) if len(set(np.round(pt, 9))) > 1 else .5
        anb = roc_auc_score(yte[nb], np.asarray(pt)[nb]) if len(set(np.round(np.asarray(pt)[nb], 9))) > 1 else .5
        bal = bal5050(pt, tau if tau is not None else tau_val(pv)) if (pv is not None or tau is not None) else float("nan")
        bal05 = bal5050(pt, 0.5)   # threshold-fixed variant (comparable across all forecasters)
        rows.append((name, auc, anb, brier_score_loss(yte, pt), brier_score_loss(yte[nb], np.asarray(pt)[nb]), bal, bal05))

    pv, pt = fc(["a", "g", "B", "base"], G); add("BEST: {α, γ_SC, B, base} (self-consist γ)", pt, pv)
    pv, pt = fc(["a"], G); add("forecaster: α-only (FM base rate feat)", pt, pv)
    gv = np.array([gem["val"].get((r[1], r[2]), gm) for r in VA]); gt = np.array([gem["test"].get((r[1], r[2]), gm) for r in TE])
    add("vanilla LLM: gemini-2.5-pro (raw P)", gt, gv)
    add("vanilla LLM: gpt-5.5 (raw P)", np.array([g55.get((r[1], r[2]), gm) for r in TE]))
    add("baseline: per-FM base rate α (raw)", np.array([alpha[r[2]] for r in TE]), np.array([alpha[r[2]] for r in VA]))
    add("baseline: weak-model transfer (gpt-4o-mini)", np.array([weak.get((r[1], r[2]), gm) for r in TE]), np.array([weak.get((r[1], r[2]), gm) for r in VA]))
    add("baseline: constant 0.5", np.full(len(TE), .5), tau=.5)
    add("baseline: constant 0.25", np.full(len(TE), .25), tau=.25)
    add("baseline: always 'not emerged' (0)", np.full(len(TE), 0.), tau=.5)

    print(f"\n3-way TEST: {len(TE)} cells ({len(nb)} non-benign); emerged {int(yte.sum())} ({yte.mean():.1%})\n")
    print(f"{'forecaster / baseline':46s}{'AUROC':>8s}{'AUROC_nb':>10s}{'Brier':>8s}{'Brier_nb':>10s}{'bal(val)':>10s}{'bal@.5':>9s}")
    print("-" * 101)
    for n, auc, anb, br, brn, bal, bal05 in rows:
        bs = f"{bal:.3f}" if bal == bal else "  —"
        print(f"{n:46s}{auc:>8.3f}{anb:>10.3f}{br:>8.3f}{brn:>10.3f}{bs:>10s}{bal05:>9.3f}")
    print("\n(γ_SC = self-consistent coherence γ, robust K=10. bal50/50 threshold tuned on val. AUROC_nb/Brier_nb = non-benign only.)")


if __name__ == "__main__":
    main()
