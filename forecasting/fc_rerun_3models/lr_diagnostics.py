import sys, numpy as np
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; FC = HERE.parent
sys.path.insert(0, str(FC)); sys.path.insert(0, str(FC/"final_system"))
import features as ict, final_scorecard as FSC
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from scipy import stats
from sklearn.metrics import roc_auc_score

build, alpha, gm = ict.build_features()
TR, VA = build("train"), build("val")
gem = {s: ict.gemini_gamma(f"broadem_google_gemini25pro{'' if s=='test' else '_'+s}.jsonl") for s in ["train","val"]}
G = {"train": {**gem["train"], **FSC.sc_load("gamma_sc_train.jsonl")},
     "val":   {**gem["val"],   **FSC.sc_load("gamma_sc_val.jsonl")}}
def Bof(g):
    b=defaultdict(float)
    for (d,f),v in g.items(): b[d]=max(b[d],v)
    return b
def feats(split, rec):
    g,B=G[split],Bof(G[split])
    return np.array([[alpha[r[2]], g.get((r[1],r[2]),gm), B.get(r[1],0.), r[3]["base"]] for r in rec],float)
# Use train cells (the fit reported in the paper)
X = feats("train", TR); y=np.array([r[4] for r in TR],float)
names=["alpha","gamma","B","b"]
n=len(y); print(f"n={n}  positives={int(y.sum())}  base_rate={y.mean():.3f}  EPV={y.sum()/4:.1f}")
print("feature ranges:", {names[j]:(round(X[:,j].min(),3),round(X[:,j].max(),3)) for j in range(4)})

# standardize (as model does)
mu,sd=X.mean(0),X.std(0)+1e-9; Z=(X-mu)/sd
Zc=sm.add_constant(Z)
m=sm.Logit(y,Zc).fit(disp=0)
print("\n=== Wald / coefficients (standardized) ===")
print(f"{'term':8s}{'coef':>8s}{'se':>7s}{'z':>7s}{'p':>9s}{'OR':>8s}  95%CI(OR)")
ci=m.conf_int()
for i,t in enumerate(["const"]+names):
    orr=np.exp(m.params[i]); lo,hi=np.exp(ci[i])
    print(f"{t:8s}{m.params[i]:8.3f}{m.bse[i]:7.3f}{m.tvalues[i]:7.2f}{m.pvalues[i]:9.2e}{orr:8.2f}  [{lo:.2f},{hi:.2f}]")
print(f"\nMcFadden R2 = {m.prsquared:.3f}   LLR p (vs intercept) = {m.llr_pvalue:.2e}   converged={m.mle_retvals['converged']}  max|coef|={np.abs(m.params).max():.2f}")

print("\n=== Likelihood-ratio drop-one tests ===")
full=m.llf
for j in range(4):
    Zr=np.delete(Z,j,axis=1); mr=sm.Logit(y,sm.add_constant(Zr)).fit(disp=0)
    lr=2*(full-mr.llf); p=stats.chi2.sf(lr,1)
    print(f"drop {names[j]:6s}: LR chi2(1)={lr:7.2f}  p={p:.2e}")

print("\n=== Multicollinearity ===")
for j in range(4): print(f"VIF {names[j]:6s} = {VIF(np.asarray(Zc), j+1):.2f}")
print("condition number =", f"{np.linalg.cond(Z):.2f}")
print("corr matrix:")
C=np.corrcoef(X.T)
print("        "+" ".join(f"{t:>7s}" for t in names))
for i,t in enumerate(names): print(f"{t:6s} "+" ".join(f"{C[i,k]:7.2f}" for k in range(4)))

print("\n=== Linearity of logit (Box-Tidwell: add x*ln x) ===")
Xp=X.copy(); Xp[:,3]=np.clip(Xp[:,3],1e-4,None); Xp=np.clip(Xp,1e-4,None)
bt_cols=[Xp[:,j]*np.log(Xp[:,j]) for j in range(4)]
Zbt=sm.add_constant(np.column_stack([Z]+bt_cols))
mbt=sm.Logit(y,Zbt).fit(disp=0)
for j in range(4):
    idx=1+4+j; print(f"  {names[j]:6s} x*ln x term: p={mbt.pvalues[idx]:.3f}  ({'nonlinear' if mbt.pvalues[idx]<0.05 else 'ok/linear'})")

print("\n=== Goodness of fit: Hosmer-Lemeshow (10 groups) ===")
p_hat=m.predict(Zc); order=np.argsort(p_hat); g=10
grp=np.array_split(order,g); hl=0.
for gi in grp:
    o1=y[gi].sum(); e1=p_hat[gi].sum(); nn=len(gi)
    o0=nn-o1; e0=nn-e1
    if e1>0: hl+=(o1-e1)**2/e1
    if e0>0: hl+=(o0-e0)**2/e0
print(f"HL chi2={hl:.2f}  df={g-2}  p={stats.chi2.sf(hl,g-2):.3f}  ({'fits' if stats.chi2.sf(hl,g-2)>0.05 else 'misfit'})")

print("\n=== Influential points ===")
infl=m.get_influence()
cook=infl.cooks_distance[0]; hat=infl.hat_matrix_diag
thr=4/n; print(f"Cook's D > 4/n ({thr:.4f}): {int((cook>thr).sum())} points ({100*(cook>thr).mean():.1f}%)  max={cook.max():.3f}")
print(f"leverage > 3*mean: {int((hat>3*hat.mean()).sum())} points")

print("\n=== Independence: cluster-robust SEs (cluster by dataset) ===")
ds=np.array([r[1] for r in TR])
mc=sm.Logit(y,Zc).fit(disp=0,cov_type='cluster',cov_kwds={'groups':ds})
print(f"{'term':8s}{'coef':>8s}{'se_iid':>8s}{'se_clu':>8s}{'p_clu':>10s}")
for i,t in enumerate(["const"]+names):
    print(f"{t:8s}{m.params[i]:8.3f}{m.bse[i]:8.3f}{mc.bse[i]:8.3f}{mc.pvalues[i]:10.2e}")

print("\n=== Drop-one ΔAUROC (importance on the headline metric, in-sample) ===")
base_auc=roc_auc_score(y,m.predict(Zc))
print(f"full in-sample AUROC={base_auc:.3f}")
for j in range(4):
    Zr=np.delete(Z,j,axis=1); mr=sm.Logit(y,sm.add_constant(Zr)).fit(disp=0)
    a=roc_auc_score(y,mr.predict(sm.add_constant(Zr)))
    print(f"  drop {names[j]:6s}: AUROC={a:.3f}  dAUROC={base_auc-a:+.3f}")
