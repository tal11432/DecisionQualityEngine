"""Gate G2-cal for the target-free hazard model (PREREG_DQ_v2.txt)."""
import sys, os; D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, pandas as pd
from tfcommon import FEATS
rng = np.random.default_rng(0)
S = pd.read_pickle(f"{D}/steps_sb.pkl"); Pi = pd.read_pickle(f"{D}/passes_sb.pkl"); M = pd.read_pickle(f"{D}/steps_metrica.pkl")
def fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1]); R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    b[0] = np.log(y.mean() / (1 - y.mean()))
    for _ in range(100):
        p = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (y - p) - R @ b); b += s
        if np.abs(s).max() < 1e-9: break
    f = lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b))); f.b = b; f.mu = mu; f.sd = sd; return f
def ll(y, p): p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def slope(y, p):
    lp = np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12))); Z = np.column_stack([np.ones(len(lp)), lp]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def report(name, y, p, base, grp, gate_tau=True):
    gain = ll(y, np.full(len(y), base)) - ll(y, p); ug = np.unique(grp); idx = {u: np.flatnonzero(grp == u) for u in ug}
    bs = [gain[np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])].sum() / len(y) for _ in range(1000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    sl = slope(y, p); oe = y.sum() / p.sum()
    O = np.array([y[idx[u]].sum() for u in ug]); E = np.array([p[idx[u]].sum() for u in ug]); tau = np.sqrt(max(0.0, np.var(np.log((O + 0.5) / E), ddof=1) - np.mean(1 / E)))
    q = pd.qcut(p, 5, labels=False, duplicates="drop"); rel = ", ".join(f"{p[q==k].mean():.4f}->{y[q==k].mean():.4f}" for k in np.unique(q))
    ok = lo > 0 and 0.8 <= sl <= 1.2 and 0.85 <= oe <= 1.15 and (tau <= 0.2 or not gate_tau)
    print(f"  {name}: step rows {len(y)}, events {int(y.sum())}; LL gain/row {gain.mean():+.5f} [{lo:+.5f},{hi:+.5f}]; slope {sl:.3f}; O/E {oe:.3f}; tau {tau:.3f} over {len(ug)} units; O/E per unit range {np.min(O/E):.2f}-{np.max(O/E):.2f}  => {'PASS' if ok else 'FAIL'}")
    print(f"     reliability (5 bins) step hazard pred->obs: {rel}"); return ok
X = S[FEATS].to_numpy(float); y = S.event.to_numpy(); comp = S.comp.to_numpy(); match = S.match.to_numpy()
print(f"StatsBomb: passes {len(Pi)} (failed {Pi.failed.sum()}, {Pi.failed.mean()*100:.1f}%), step rows {len(S)}")
oks = []
for tr, te in (("EU20", "WC22"), ("WC22", "EU20")):
    a, b_ = comp == tr, comp == te; f = fit(X[a], y[a]); print(f"\ntrain {tr} -> test {te}"); oks.append(report(te, y[b_], f(X[b_]), y[a].mean(), match[b_]))
fa = fit(X, y); print("\ncoefficients (standardised, both competitions): " + ", ".join(f"{n} {c:+.2f}" for n, c in zip(["const"] + FEATS, fa.b)))
Xm = M[FEATS].to_numpy(float); ym = M.event.to_numpy(); pm = fa(Xm)
print("\nTransfer to Metrica (fitted on EU20+WC22):")
okm = report("Metrica pooled (units = games)", ym, pm, y.mean(), M.game.to_numpy(), gate_tau=False)
for g in (1, 2, 3):
    m = M.game.to_numpy() == g; print(f"   G{g}: events {int(ym[m].sum())}, O/E {ym[m].sum()/pm[m].sum():.3f}, slope {slope(ym[m], pm[m]):.3f}")
# pass-level illustration (not gated): predicted P(survive to observed length) for completed passes
pid = M.pid.to_numpy(); surv = pd.Series(np.log1p(-pm)).groupby(pid).sum().pipe(np.exp)
print(f"\nGATE G2-cal: cross-competition {'PASS' if all(oks) else 'FAIL'}; Metrica transfer {'PASS' if okm else 'FAIL'}  => {'PASS' if all(oks) and okm else 'FAIL'}")
pd.to_pickle(dict(b=fa.b, mu=fa.mu, sd=fa.sd, feats=FEATS), f"{D}/tf_model.pkl")
