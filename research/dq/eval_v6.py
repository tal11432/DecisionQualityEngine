"""DQ-v6 analysis per PREREG_DQ_v6.txt."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS as HF
rng = np.random.default_rng(0); GX = 120.0; SC = np.array([105 / 120, 68 / 80])
Sh = pd.read_pickle(f"{D}/v6_shots.pkl"); Pa = pd.read_pickle(f"{D}/v6_passes.pkl"); M5 = pd.read_pickle(f"{D}/v5_models.pkl"); HM = pd.read_pickle(f"{D}/tf_model.pkl"); XG0 = pd.read_pickle(f"{D}/stage1b_models.pkl")["xG"]
Sh["da"], Sh["ldist"] = Sh.dist * Sh.angle, np.log(Sh.dist); Pa["d2"], Pa["ad"], Pa["x"] = Pa.dist ** 2, Pa.angle * Pa.dist, Pa.tx
def fit(X, y, lam=1.0, it=300, clip=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1]); R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    b[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b)); s = np.clip(np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (y - p) - R @ b), -clip, clip); b += s
        if np.abs(s).max() < 1e-9: break
    f = lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b))); f.coef = dict(mu=mu, sd=sd, b=b); return f
ll = lambda y, p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(1 - np.clip(p, 1e-9, 1 - 1e-9)))
def slope(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); Z = np.column_stack([np.ones(len(lp)), lp]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def mboot(v, mt):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(500)]; return np.percentile(b, [2.5, 97.5])
FA5 = M5["xA"]["F"]; FA6 = FA5 + ["fwd", "pdist"]
ptr, pte = Pa[~Pa.test].reset_index(drop=True), Pa[Pa.test].reset_index(drop=True)
print(f"passes: train {len(ptr)} (mean v_recv {ptr.v_recv.mean():.4f}, mean v_chain {ptr.v_chain.mean():.4f}), test {len(pte)} (v_recv {pte.v_recv.mean():.4f}, v_chain {pte.v_chain.mean():.4f}); test comps {dict(pte.comp.value_counts())}")
yc = np.clip(ptr.v_chain.to_numpy(float), 0, 0.99); fc_ = fit(ptr[FA6].to_numpy(float), yc); fr_ = fit(ptr[FA6].to_numpy(float), np.clip(ptr.v_recv.to_numpy(float), 0, 0.99))
hyp = lambda d: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(d)), (np.column_stack([d.dist, d.angle, np.zeros(len(d)), d.n_tri, d.gk_d]) - XG0["mu"]) / XG0["sd"]]) @ XG0["b"])))
base = yc.mean(); ok = True; slopes = []
for lab, d in [("POOLED", pte)] + [(c, pte[pte.comp == c]) for c in sorted(pte.comp.unique())]:
    y = np.clip(d.v_chain.to_numpy(float), 0, 0.99); p = fc_(d[FA6].to_numpy(float)); g = ll(y, np.full(len(y), base)) - ll(y, p); lo, hi = mboot(g, d.match.to_numpy()); sl = slope(y, p)
    print(f"[{lab}] n={len(d)}: observed chain value {y.mean():.4f}; predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {sl:.3f}, LL gain vs constant {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}]); receiver-only value {d.v_recv.mean():.4f}; old hypothetical xA {hyp(d).mean():.4f}")
    if lab == "POOLED": ok = lo > 0 and 0.8 <= sl <= 1.2 and 0.85 <= p.mean() / y.mean() <= 1.15
    elif len(d) >= 150: slopes.append(sl)
okc = ok and all(0.6 <= s <= 1.4 for s in slopes); print(f"GATE G-xAc: {'PASS' if okc else 'FAIL'} (per-competition slopes {[round(s, 2) for s in slopes]})")
by = pd.cut(pte.dist * 105 / 120, [0, 12, 20, 30, 80]); print("by target distance (m): " + "; ".join(f"{k}: observed chain {g.v_chain.mean():.3f} | predicted {fc_(g[FA6].to_numpy(float)).mean():.3f} | receiver-only {g.v_recv.mean():.3f} | hypothetical {hyp(g).mean():.3f} (n={len(g)})" for k, g in pte.groupby(by, observed=True)))
# continuation of shots
str_, ste = Sh[~Sh.test].reset_index(drop=True), Sh[Sh.test].reset_index(drop=True); F1 = ["dist", "angle", "header", "n_tri", "gk_d"]
fcont = fit(str_[F1].to_numpy(float), np.clip(str_.cont.to_numpy(float), 0, 0.99)); y = np.clip(ste.cont.to_numpy(float), 0, 0.99); p = fcont(ste[F1].to_numpy(float))
print(f"\nshot continuation c (later shots in the possession within 10 s): train {len(str_)} shots mean {str_.cont.mean():.4f}; test {len(ste)} observed {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}); shot xG mean {ste.sb_xg.mean():.4f}")
print(f"GATE G-c: {'PASS' if 0.85 <= p.mean() / y.mean() <= 1.15 else 'FAIL'}")
# shot vs pass
def hz(a, end, O):
    R = steps(a, end, O, False, None); lg = np.column_stack([np.ones(len(R)), (R[:, :len(HF)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-lg))).sum()))
def tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
sgn = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
fs = ste[ste.header == 0].reset_index(drop=True); geo = M5["xG_geo"]; fs["xg"] = [sgn(geo, r[geo["F"]].to_numpy(float)) for _, r in fs.iterrows()]; fs["cont_hat"] = fcont(fs[F1].to_numpy(float)); rows = []
for _, r in fs.iterrows():
    a = np.array([r.x, r.y]); O = np.array(r.opps, float).reshape(-1, 2); gk = np.array(r.gk); ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); b_chain = b_recv = b_hyp = 0.0
    for p in r.mates:
        p = np.array(p, float); d = float(np.hypot(*((p - a) * SC)))
        if not (3 <= d <= 40) or (p[0] > line and p[0] > 60): continue
        dd = np.hypot(*(O - p).T); dist = float(np.hypot(GX - p[0], 40 - p[1])); an = float(abs(np.arctan2(44 - p[1], GX - p[0]) - np.arctan2(36 - p[1], GX - p[0]))); nt = sum(tri(o, p) for o in O); gd = float(np.hypot(GX - gk[0], 40 - gk[1]))
        base5 = [dist, an, p[0], dist ** 2, an * dist, nt, (dd <= 3).sum(), (dd <= 6).sum(), dd.min(), gd, float(np.hypot(*(p - a)))]
        xa_c = fc_(np.array([base5 + [p[0] - a[0], float(np.hypot(GX - a[0], 40 - a[1]))]]))[0]; xa_r = sgn(M5["xA"], base5)
        h = float(1 / (1 + np.exp(-(np.array([1, *((np.array([dist, an, 0, nt, gd]) - XG0["mu"]) / XG0["sd"])]) @ XG0["b"])))); xp = hz(a, p, O)
        b_chain, b_recv, b_hyp = max(b_chain, xp * xa_c), max(b_recv, xp * xa_r), max(b_hyp, xp * h)
    rows.append((b_chain, b_recv, b_hyp))
fs["pass_chain"], fs["pass_recv"], fs["pass_hyp"] = zip(*rows); fs["shot_val"] = fs.xg + fs.cont_hat; fs["band"] = pd.cut(fs.dist * 105 / 120, [0, 12, 20, 200], labels=["<12m", "12-20m", ">20m"])
print(f"\nSHOT vs PASS on {len(fs)} test foot shots (shot value = xG_geo [+ continuation for the chain comparison]):")
print(f"  lower bound (receiver-only xA vs xG):        better pass {(fs.pass_recv > fs.xg).mean()*100:.1f}%")
print(f"  PRIMARY (possession chain vs xG + continuation): better pass {(fs.pass_chain > fs.shot_val).mean()*100:.1f}%, median gap {(fs.pass_chain-fs.shot_val).median():+.4f}, sum shot {fs.shot_val.sum():.1f} vs sum best pass {fs.pass_chain.sum():.1f}; gap>0.05 in {((fs.pass_chain-fs.shot_val)>0.05).mean()*100:.1f}%")
ratio = float(np.clip(pte.v_chain, 0, 0.99).mean() / fc_(pte[FA6].to_numpy(float)).mean()); print(f"  DIAGNOSTIC (uses test labels, not a prediction): chain value rescaled by observed/predicted = {ratio:.3f}: better pass {(fs.pass_chain*ratio > fs.shot_val).mean()*100:.1f}%, median gap {(fs.pass_chain*ratio-fs.shot_val).median():+.4f}; by band " + ", ".join(f"{b} {(g.pass_chain*ratio > g.shot_val).mean()*100:.0f}%" for b, g in fs.groupby("band", observed=True)))
print(f"  (chain vs xG alone, no shot continuation):    better pass {(fs.pass_chain > fs.xg).mean()*100:.1f}%")
print(f"  upper bound (hypothetical xA vs xG):          better pass {(fs.pass_hyp > fs.xg).mean()*100:.1f}%")
for b, g in fs.groupby("band", observed=True): print(f"  {b}: n={len(g)} primary {(g.pass_chain > g.shot_val).mean()*100:.1f}% | receiver-only {(g.pass_recv > g.xg).mean()*100:.1f}% | hypothetical {(g.pass_hyp > g.xg).mean()*100:.1f}%; mean shot value {g.shot_val.mean():.3f} vs best pass {g.pass_chain.mean():.3f}")
for c, g in fs.groupby("comp"): print(f"  {c}: n={len(g)} primary {(g.pass_chain > g.shot_val).mean()*100:.1f}%")
pd.to_pickle(dict(xA_chain=dict(F=FA6, **fc_.coef), cont=dict(F=F1, **fcont.coef)), f"{D}/v6_models.pkl")
