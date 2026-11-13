"""Stage 1: V, Vopp, xG fitted on PL 2015/16 first half; tested on PL second half (A) and WC 2018 sample (B). Gate G1 frozen in PREREG_DQ_v0.txt."""
import numpy as np, pandas as pd, os
D = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
meta = pd.read_pickle(f"{D}/meta.pkl"); ob = pd.read_pickle(f"{D}/onball.pkl"); st = pd.read_pickle(f"{D}/starts.pkl"); sh = pd.read_pickle(f"{D}/shots.pkl"); M = pd.read_pickle(f"{D}/metrica_shots.pkl")
pl = meta[meta.comp == "PL"]; cut = pl.date.sort_values().iloc[len(pl) // 2]
split = {r.match: ("train" if (r.comp == "PL" and r.date < cut) else "A" if r.comp == "PL" else "B") for r in meta.itertuples()}
print(f"matches: train {sum(v=='train' for v in split.values())}, test A (PL after {cut}) {sum(v=='A' for v in split.values())}, test B (WC 2018) {sum(v=='B' for v in split.values())}")
GX = 120.0
def ang(x, y): return np.abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x))
def basis(d):
    dist = np.hypot(GX - d.x, 40 - d.y); a = ang(d.x, d.y)
    return np.column_stack([dist, a, d.x, dist ** 2, a * dist])
def fit(X, y, lam=1.0, it=100):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1])
    R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b)); g = Z.T @ (y - p) - R @ b; Hm = (Z * (p * (1 - p))[:, None]).T @ Z + R; s = np.linalg.solve(Hm, g); b += s
        if np.abs(s).max() < 1e-10: break
    return lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b)))
def ll(y, p): p = np.clip(p, 1e-9, 1 - 1e-9); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def slope(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); f = fit(lp[:, None], y, lam=0.0)
    # recover slope: fit returns standardized model; refit manually
    Z = np.column_stack([np.ones(len(lp)), lp]); b = np.zeros(2)
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def auc(y, p):
    r = pd.Series(p).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def evaluate(name, d, p, base, extra=None):
    out = {}
    for part in ("A", "B"):
        m = d.split.to_numpy() == part; y = d.goal.to_numpy()[m]; pp = p[m]; mt = d.match.to_numpy()[m]
        dl = ll(y, np.full(len(y), base)) - ll(y, pp)                       # positive = model better
        um = np.unique(mt); grp = {u: np.flatnonzero(mt == u) for u in um}; bs = []
        for _ in range(1000):
            ix = np.concatenate([grp[u] for u in rng.choice(um, len(um))]); bs.append(dl[ix].mean())
        sl, citl = slope(y, pp), pp.mean() / y.mean()
        row = dict(n=len(y), goals=int(y.sum()), rate=y.mean(), ll_gain=dl.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), slope=sl, citl=citl)
        if extra: row.update(extra(y, pp, m))
        ok = row["ci_lo"] > 0 and 0.8 <= sl <= 1.2 and 0.85 <= citl <= 1.15 and (row.get("auc", 1) >= 0.75)
        row["pass"] = ok; out[part] = row
        print(f"  {name} test {part}: n={len(y)} events, {int(y.sum())} positives (rate {y.mean():.4f}); log-loss gain vs constant {dl.mean():+.5f} [{row['ci_lo']:+.5f},{row['ci_hi']:+.5f}]; slope {sl:.3f}; mean p/mean y {citl:.3f}"
              + (f"; AUC {row['auc']:.3f} (StatsBomb xG {row['sb_auc']:.3f}, log loss ours {row['ll']:.4f} vs StatsBomb {row['sb_ll']:.4f})" if extra else "") + f"  => {'PASS' if ok else 'FAIL'}")
        # deciles
        q = pd.qcut(pp, 5, labels=False, duplicates="drop"); print("     reliability (5 bins) pred->obs: " + ", ".join(f"{pp[q==k].mean():.3f}->{y[q==k].mean():.3f}" for k in np.unique(q)))
    return out
res = {}
for nm, d in (("V (possession ends in goal, from any on-ball event)", ob), ("Vopp (from first event of possessions after open-play turnover)", st[st.pattern.isin(["Regular Play", "From Counter"])])):
    d = d.copy(); d["split"] = d.match.map(split); tr = d.split == "train"; X = basis(d); f = fit(X[tr.to_numpy()], d.goal.to_numpy()[tr.to_numpy()])
    p = f(X); base = d.goal[tr].mean(); print(f"\n{nm}: train n={tr.sum()} rate {base:.4f}"); res[nm[:4]] = evaluate(nm[:4], d, p, base)
    if nm.startswith("V "): Vf = f
    else: Vof = f
# xG
d = sh.copy(); d["split"] = d.match.map(split); d = d.dropna(subset=["gk_d"]); tr = (d.split == "train").to_numpy()
XF = lambda q: np.column_stack([q.dist, q.angle, q.header, q.n_tri, q.gk_d]); fx = fit(XF(d)[tr], d.goal.to_numpy()[tr]); p = fx(XF(d)); base = d.goal[tr].mean()
print(f"\nxG (open play, non-penalty): train n={tr.sum()} goals {d.goal[tr].sum()}")
res["xG"] = evaluate("xG", d, p, base, extra=lambda y, pp, m: dict(auc=auc(y, pp), sb_auc=auc(y, d.sb_xg.to_numpy()[m]), ll=ll(y, pp).mean(), sb_ll=ll(y, d.sb_xg.to_numpy()[m]).mean()))
# Metrica transfer check (weak): Poisson 95% interval of goals given predicted xG sum
pm = fx(XF(M)); lam = pm.sum(); k = M.goal.sum()
from math import exp, factorial
cdf = lambda kk: sum(exp(-lam) * lam ** i / factorial(i) for i in range(kk + 1))
lo = next(i for i in range(200) if cdf(i) >= 0.025); hi = next(i for i in range(200) if cdf(i) >= 0.975)
mp = (lo <= k <= hi); print(f"\nMetrica transfer: 50 open-play foot shots, predicted xG sum {lam:.2f} (StatsBomb-trained), Poisson 95% range [{lo},{hi}], observed goals {k} => {'PASS' if mp else 'FAIL'}; per-shot xG median {np.median(pm):.3f}, max {pm.max():.3f}")
print("by game: " + ", ".join(f"G{g}: xG {pm[M.game.to_numpy()==g].sum():.2f} vs goals {M.goal[M.game==g].sum()}" for g in (1, 2, 3)))
G1 = all(r["pass"] for v in res.values() for r in v.values()) and mp
print(f"\nGATE G1: {'PASS' if G1 else 'FAIL'}")
pd.to_pickle(dict(res=res, metrica_xg=pm), f"{D}/stage1_res.pkl")
