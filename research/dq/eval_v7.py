"""DQ-v7 analysis per PREREG_DQ_v7.txt."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS as HF
rng = np.random.default_rng(0); GX = 120.0; SC = np.array([105 / 120, 68 / 80])
Sh = pd.read_pickle(f"{D}/v7_shots.pkl"); Pa = pd.read_pickle(f"{D}/v7_passes.pkl"); M5 = pd.read_pickle(f"{D}/v5_models.pkl"); HM = pd.read_pickle(f"{D}/tf_model.pkl"); XG0 = pd.read_pickle(f"{D}/stage1b_models.pkl")["xG"]
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

Fa = pd.read_pickle(f"{D}/v7_fails.pkl")
# ---------- (1) level variability across competitions (LOCO)
pa = Pa.copy(); rows = []
for c in sorted(pa.comp.unique()):
    d = pa[pa.comp == c]
    if len(d) < 150: continue
    o = pa[pa.comp != c]; f = fit(o[FA6].to_numpy(float), np.clip(o.v_chain.to_numpy(float), 0, 0.99)); y = np.clip(d.v_chain.to_numpy(float), 0, 0.99)
    pos = int((d.v_chain > 0).sum()); rows.append((c, len(d), pos, y.mean(), f(d[FA6].to_numpy(float)).mean())); 
L = pd.DataFrame(rows, columns=["comp", "n", "pos", "obs", "pred"]); L["r"] = L.obs / L.pred
print("\nLEVEL VARIABILITY of the chain value across competitions (each predicted by a model fit on all others):")
for _, r in L.iterrows(): print(f"   {r.comp:<28} n={r.n:>5} passes followed by a chain shot {r.pos:>4}  observed {r.obs:.4f}  predicted {r.pred:.4f}  ratio {r.r:.3f}")
v = np.var(np.log(L.r), ddof=1); noise = np.mean(1 / L.pos); tau = float(np.sqrt(max(0.0, v - noise)))
print(f"   var(log ratio) {v:.4f}, expected sampling noise {noise:.4f} -> tau = {tau:.3f}; 95% range of the level multiplier exp(+-1.96 tau) = [{np.exp(-1.96*tau):.2f}, {np.exp(1.96*tau):.2f}]")
# ---------- (2) failure cost model
Fa["d2"] = 0.0
def basis_xy(x, y):
    dist = np.hypot(GX - x, 40 - y); an = np.abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)); return np.column_stack([dist, an, x, dist ** 2, an * dist])
ftr, fte = Fa[~Fa.test], Fa[Fa.test]; fvf = fit(basis_xy(ftr.fx.to_numpy(float), ftr.fy.to_numpy(float)), np.clip(ftr.vf.to_numpy(float), 0, 0.99))
y = np.clip(fte.vf.to_numpy(float), 0, 0.99); p = fvf(basis_xy(fte.fx.to_numpy(float), fte.fy.to_numpy(float))); g = ll(y, np.full(len(y), np.clip(ftr.vf, 0, .99).mean())) - ll(y, p); lo, hi = mboot(g, fte.match.to_numpy())
print(f"\nFAILURE COST: failed passes train {len(ftr)} (mean opponent chain xG {ftr.vf.mean():.4f}), test {len(fte)} (observed {y.mean():.4f}, predicted {p.mean():.4f}, ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}, LL gain {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}]); opponent shots after {int((fte.vf>0).sum())} test failures")
okv = lo > 0 and 0.85 <= p.mean() / y.mean() <= 1.15; print(f"GATE G-vf: {'PASS' if okv else 'FAIL'}")
by = pd.cut(fte.fx * 105 / 120, [0, 20, 40, 60, 120]); print("   by distance of the loss point to the OPPONENT's goal (m): " + "; ".join(f"{k}: observed {g2.vf.mean():.4f} predicted {fvf(basis_xy(g2.fx.to_numpy(float), g2.fy.to_numpy(float))).mean():.4f} (n={len(g2)})" for k, g2 in fte.assign(b=by).groupby("b", observed=True)))
# ---------- shot vs pass with failure cost, level robustness, three-way classification
def hz(a, end, O):
    R = steps(a, end, O, False, None); lg = np.column_stack([np.ones(len(R)), (R[:, :len(HF)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-lg))).sum()))
def tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
sgn = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
fs = ste[ste.header == 0].reset_index(drop=True); geo = M5["xG_geo"]; fs["xg"] = [sgn(geo, r[geo["F"]].to_numpy(float)) for _, r in fs.iterrows()]; fs["cont_hat"] = fcont(fs[F1].to_numpy(float)); rows = []
for _, r in fs.iterrows():
    a = np.array([r.x, r.y]); O = np.array(r.opps, float).reshape(-1, 2); gk = np.array(r.gk); ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); bc = 0.0; bn = -1.0
    for p in r.mates:
        p = np.array(p, float); d = float(np.hypot(*((p - a) * SC)))
        if not (3 <= d <= 40) or (p[0] > line and p[0] > 60): continue
        dd = np.hypot(*(O - p).T); dist = float(np.hypot(GX - p[0], 40 - p[1])); an = float(abs(np.arctan2(44 - p[1], GX - p[0]) - np.arctan2(36 - p[1], GX - p[0]))); nt = sum(tri(o, p) for o in O); gd = float(np.hypot(GX - gk[0], 40 - gk[1]))
        xa_c = fc_(np.array([[dist, an, p[0], dist ** 2, an * dist, nt, (dd <= 3).sum(), (dd <= 6).sum(), dd.min(), gd, float(np.hypot(*(p - a))), p[0] - a[0], float(np.hypot(GX - a[0], 40 - a[1]))]]))[0]
        xp = hz(a, p, O); vf = fvf(basis_xy(np.array([120.0 - p[0]]), np.array([80.0 - p[1]])))[0]
        bc = max(bc, xp * xa_c); bn = max(bn, xp * xa_c - (1 - xp) * vf)
    rows.append((bc, max(bn, 0.0) if bn > -1 else 0.0, bn > -1))
fs["pass_chain"], fs["pass_net"], fs["has_cand"] = zip(*rows); fs["shot_val"] = fs.xg + fs.cont_hat; fs["band"] = pd.cut(fs.dist * 105 / 120, [0, 12, 20, 200], labels=["<12m", "12-20m", ">20m"])
fs = fs[fs.has_cand].reset_index(drop=True)
print(f"\nSHOT vs PASS, {len(fs)} test foot shots with at least one admissible candidate:")
print(f"  chain value, no failure cost : better pass {(fs.pass_chain > fs.shot_val).mean()*100:.1f}%  | with failure cost: {(fs.pass_net > fs.shot_val).mean()*100:.1f}%  (mean failure cost of the best pass: {(fs.pass_chain-fs.pass_net).mean():.5f}; mean shot value {fs.shot_val.mean():.3f})")
print("  ROBUSTNESS CURVE (share with pass > shot when the chain value is multiplied by s): " + ", ".join(f"s={s:.1f}: {(fs.pass_net*s > fs.shot_val).mean()*100:.0f}%" for s in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)))
def cls(d, lo=0.7, hi=1.3): return (d.pass_net * lo > d.shot_val), (d.pass_net * hi < d.shot_val)
def three(d, lo=0.7, hi=1.3):
    pc, sc = cls(d, lo, hi); return pc.mean() * 100, sc.mean() * 100, (1 - pc.mean() - sc.mean()) * 100
def tci(d, which, lo=0.7, hi=1.3):
    um = d.match.unique() if "match" in d else None; g = {u: np.flatnonzero(d.match.to_numpy() == u) for u in um}; v = np.array([cls(d, lo, hi)[0].to_numpy(), cls(d, lo, hi)[1].to_numpy()])
    b = []
    for _ in range(300): ix = np.concatenate([g[u] for u in rng.choice(um, len(um))]); b.append(v[:, ix].mean(1))
    b = np.array(b); return np.percentile(b[:, which], [2.5, 97.5]) * 100
pc, sc, am = three(fs); print(f"  THREE-WAY (s in [0.7,1.3]): pass clearly better {pc:.1f}% [{tci(fs, 0)[0]:.1f},{tci(fs, 0)[1]:.1f}], shot clearly better {sc:.1f}% [{tci(fs, 1)[0]:.1f},{tci(fs, 1)[1]:.1f}], ambiguous {am:.1f}%")
if tau > 0.3: lo2, hi2 = np.exp(-1.96 * tau), np.exp(1.96 * tau); pc2, sc2, am2 = three(fs, lo2, hi2); print(f"  THREE-WAY (s in [{lo2:.2f},{hi2:.2f}] from tau): pass clearly better {pc2:.1f}%, shot clearly better {sc2:.1f}%, ambiguous {am2:.1f}%")
for b, g in fs.groupby("band", observed=True): a1, a2, a3 = three(g); print(f"    {b}: n={len(g)} pass clearly better {a1:.1f}%, shot clearly better {a2:.1f}%, ambiguous {a3:.1f}%")
for c, g in fs.groupby("comp"): a1, a2, a3 = three(g); print(f"    {c}: n={len(g)} pass clearly better {a1:.1f}%, shot clearly better {a2:.1f}%, ambiguous {a3:.1f}%")
pd.to_pickle(dict(xA_chain=dict(F=FA6, **fc_.coef), cont=dict(F=F1, **fcont.coef), vf=dict(**fvf.coef), tau=tau), f"{D}/v7_models.pkl")
