"""Stage 1 repaired (DQ-v1). Same models as v0 (imported fitting code copied verbatim), new gates, new test competitions C=EU20, D=WC22."""
import numpy as np, pandas as pd, os
D = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
meta = pd.read_pickle(f"{D}/meta.pkl"); ob = pd.read_pickle(f"{D}/onball.pkl"); st = pd.read_pickle(f"{D}/starts.pkl"); sh = pd.read_pickle(f"{D}/shots.pkl"); M = pd.read_pickle(f"{D}/metrica_shots.pkl")
pl = meta[meta.comp == "PL"]; cut = pl.date.sort_values().iloc[len(pl) // 2]
lab = {"PL": None, "WC18": "B", "EU20": "C", "WC22": "D"}
split = {r.match: (("train" if r.date < cut else "A") if r.comp == "PL" else lab[r.comp]) for r in meta.itertuples()}
print("matches per split:", pd.Series(split).value_counts().to_dict(), "cut", cut)
GX = 120.0
def ang(x, y): return np.abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x))
def basis(d):
    dist = np.hypot(GX - d.x, 40 - d.y); a = ang(d.x, d.y); return np.column_stack([dist, a, d.x, dist ** 2, a * dist])
def fit(X, y, lam=1.0, it=100):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1]); R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (y - p) - R @ b); b += s
        if np.abs(s).max() < 1e-10: break
    f = lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b))); f.coef = dict(mu=mu, sd=sd, b=b); return f
def ll(y, p): p = np.clip(p, 1e-9, 1 - 1e-9); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def slope(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); Z = np.column_stack([np.ones(len(lp)), lp]); b = np.zeros(2)
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def auc(y, p):
    r = pd.Series(p).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def mboot(v, mt):
    um = np.unique(mt); grp = {u: np.flatnonzero(mt == u) for u in um}; bs = [v[np.concatenate([grp[u] for u in rng.choice(um, len(um))])].mean() for _ in range(1000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
d = ob.copy(); d["split"] = d.match.map(split); X = basis(d); tr = (d.split == "train").to_numpy(); V = fit(X[tr], d.goal.to_numpy()[tr]); d["p"] = V(X)
s0 = st[st.pattern.isin(["Regular Play", "From Counter"])].copy(); s0["split"] = s0.match.map(split)
k_train = s0.goal[s0.split == "train"].mean() / d.p[tr].mean()
print(f"\nREPAIR 2: k_train = turnover-possession goal rate {s0.goal[s0.split=='train'].mean():.4f} / mean V {d.p[tr].mean():.4f} = {k_train:.3f}")
Xo = basis(s0); Vo = fit(Xo[(s0.split == "train").to_numpy()], s0.goal.to_numpy()[(s0.split == "train").to_numpy()]); s0["p"] = Vo(Xo)
gates = {}
for part in ("A", "B", "C", "D"):
    m = (d.split == part).to_numpy(); y = d.goal.to_numpy()[m]; p = d.p.to_numpy()[m]; mt = d.match.to_numpy()[m]; c = y.mean() / p.mean(); pr = np.clip(p * c, 1e-9, 1 - 1e-9)
    gain = ll(y, np.full(len(y), y.mean())) - ll(y, pr); lo, hi = mboot(gain, mt); sl = slope(y, pr); q = pd.qcut(pr, 5, labels=False, duplicates="drop")
    ratios = [y[q == j].mean() / pr[q == j].mean() for j in np.unique(q)]
    okV = lo > 0 and 0.8 <= sl <= 1.2 and all(0.75 <= r <= 1.33 for r in ratios)
    ms = (s0.split == part).to_numpy(); kt = s0.goal.to_numpy()[ms].mean() / p.mean(); okK = abs(kt / k_train - 1) <= 0.25
    yo, po = s0.goal.to_numpy()[ms], s0.p.to_numpy()[ms]; go = ll(yo, np.full(len(yo), yo.mean())) - ll(yo, np.clip(po * yo.mean() / po.mean(), 1e-9, 1)); olo, ohi = mboot(go, s0.match.to_numpy()[ms])
    gates[part] = dict(V=okV, k=okK)
    print(f"\n[{part}] V: level factor c={c:.3f}; log-loss gain vs constant (level-matched) {gain.mean():+.5f} [{lo:+.5f},{hi:+.5f}]; slope {sl:.3f}; quintile obs/pred " + ", ".join(f"{r:.2f}" for r in ratios) + f"  => {'PASS' if okV else 'FAIL'}")
    print(f"     k: turnover goal rate {yo.mean():.4f} (n={len(yo)}, {yo.sum()} goals), k={kt:.3f} vs train {k_train:.3f} ({(kt/k_train-1)*100:+.0f}%)  => {'PASS' if okK else 'FAIL'};  informational: Vopp location model gain {go.mean():+.5f} [{olo:+.5f},{ohi:+.5f}]")
# xG
x = sh.dropna(subset=["gk_d"]).copy(); x["split"] = x.match.map(split); XF = lambda q: np.column_stack([q.dist, q.angle, q.header, q.n_tri, q.gk_d]); trx = (x.split == "train").to_numpy()
G = fit(XF(x)[trx], x.goal.to_numpy()[trx]); x["p"] = G(XF(x)); base = x.goal[trx].mean()
for part in ("A", "B", "C", "D"):
    m = (x.split == part).to_numpy(); y = x.goal.to_numpy()[m]; p = x.p.to_numpy()[m]; mt = x.match.to_numpy()[m]
    g = ll(y, np.full(len(y), base)) - ll(y, p); lo, hi = mboot(g, mt); sl = slope(y, p); citl = p.mean() / y.mean(); a = auc(y, p); sa = auc(y, x.sb_xg.to_numpy()[m])
    ok = lo > 0 and 0.8 <= sl <= 1.2 and 0.85 <= citl <= 1.15 and a >= 0.75; gates[part]["xG"] = ok
    print(f"[{part}] xG: n={len(y)} goals {int(y.sum())}; gain {g.mean():+.4f} [{lo:+.4f},{hi:+.4f}]; slope {sl:.3f}; mean p/mean y {citl:.3f}; AUC {a:.3f} (StatsBomb {sa:.3f})  => {'PASS' if ok else 'FAIL'}")
pm = G(XF(M)); print(f"Metrica: predicted xG {pm.sum():.2f} vs 7 goals (refit identical to v0 => should match 5.53)")
G1 = all(all(gates[p].values()) for p in ("C", "D"))
print("\ngates by test set:", gates, f"\nGATE G1' (C and D only): {'PASS' if G1 else 'FAIL'}")
pd.to_pickle(dict(V=V.coef, Vopp=Vo.coef, xG=G.coef, k_train=k_train, gates=gates), f"{D}/stage1b_models.pkl")
