"""DQ-v5 analysis per PREREG_DQ_v5.txt."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS as HF
rng = np.random.default_rng(0); GX = 120.0; SC = np.array([105 / 120, 68 / 80])
Sh = pd.read_pickle(f"{D}/v5_shots.pkl"); Pa = pd.read_pickle(f"{D}/v5_passes.pkl")
Sh = Sh.join(Sh.prior.apply(pd.Series).add_prefix("prior_")); Sh["pat_counter"] = (Sh.pattern == "counter").astype(int); Sh["pat_set"] = (Sh.pattern == "setpiece").astype(int)
Sh["da"] = Sh.dist * Sh.angle; Sh["ldist"] = np.log(Sh.dist)
XG0 = pd.read_pickle(f"{D}/stage1b_models.pkl")["xG"]; HM = pd.read_pickle(f"{D}/tf_model.pkl")
def fit(X, y, lam=1.0, it=100, clip=None):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1]); R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    b[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (y - p) - R @ b)
        if clip: s = np.clip(s, -clip, clip)
        b += s
        if np.abs(s).max() < 1e-9: break
    f = lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b))); f.coef = dict(mu=mu, sd=sd, b=b); return f
ll = lambda y, p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(1 - np.clip(p, 1e-9, 1 - 1e-9)))
def slope(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); Z = np.column_stack([np.ones(len(lp)), lp]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def auc(y, p): r = pd.Series(p).rank().to_numpy(); n1 = (y > 0.5).sum(); return (r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
def mboot(v, mt):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(500)]; return np.percentile(b, [2.5, 97.5])
F1 = ["dist", "angle", "header", "n_tri", "gk_d"]; F2 = F1 + ["gk_lat", "n_opp2", "nn_opp", "da", "ldist"]; F3 = F2 + ["first_time", "volley", "pat_counter", "pat_set", "prior_cross", "prior_through", "prior_cutback"]
tr, te = Sh[~Sh.test].reset_index(drop=True), Sh[Sh.test].reset_index(drop=True)
print(f"shots: train {len(tr)} ({tr.goal.sum()} goals, {tr.comp.nunique()} competitions), test {len(te)} ({te.goal.sum()} goals): " + ", ".join(f"{c} {n}" for c, n in te.comp.value_counts().items()))
# ---------------- (A) xG
comps = sorted(tr.comp.unique()); loco = {}
for nm, F in (("M1", F1), ("M2", F2), ("M3", F3)):
    tot = 0
    for c in comps:
        a, b = tr.comp != c, tr.comp == c
        if b.sum() < 30: continue
        f = fit(tr.loc[a, F].to_numpy(float), tr.goal[a].to_numpy(float)); tot += ll(tr.goal[b].to_numpy(float), f(tr.loc[b, F].to_numpy(float))).sum()
    loco[nm] = tot / len(tr); print(f"  LOCO log loss on train competitions {nm}: {loco[nm]:.5f}")
best = min(loco, key=loco.get); FB = {"M1": F1, "M2": F2, "M3": F3}[best]; print(f"selected xG_new = {best} ({len(FB)} features)")
Xnew = fit(tr[FB].to_numpy(float), tr.goal.to_numpy(float)); Xgeo = fit(tr[F2].to_numpy(float), tr.goal.to_numpy(float)); X1 = fit(tr[F1].to_numpy(float), tr.goal.to_numpy(float))
pred = {"M0 (old)": (lambda d: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(d)), (d[F1].to_numpy(float) - XG0["mu"]) / XG0["sd"]]) @ XG0["b"])))),
        "M1 refit": lambda d: X1(d[F1].to_numpy(float)), "xG_geo (M2)": lambda d: Xgeo(d[F2].to_numpy(float)), f"xG_new ({best})": lambda d: Xnew(d[FB].to_numpy(float)), "StatsBomb xG": lambda d: d.sb_xg.to_numpy(float)}
print("\nxG on TEST competitions (pooled and each): n, goals, O/E (pred sum / goals), slope, AUC, LL gain vs M0 [95% match-bootstrap CI]")
gate = True
for lab, d in [("POOLED", te)] + [(c, te[te.comp == c]) for c in sorted(te.comp.unique()) if (te.comp == c).sum() >= 150]:
    y = d.goal.to_numpy(float); p0 = pred["M0 (old)"](d); print(f"[{lab}] n={len(d)} goals={int(y.sum())}")
    for nm, fn in pred.items():
        p = fn(d); g = ll(y, p0) - ll(y, p); lo, hi = mboot(g, d.match.to_numpy()); print(f"   {nm:<16} pred/obs {p.sum()/y.sum():.3f}  slope {slope(y, p):.3f}  AUC {auc(y, p):.3f}  LL {ll(y, p).mean():.4f}  gain vs M0 {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}]")
        if lab == "POOLED" and nm.startswith("xG_new"): gate = lo > 0 and 0.9 <= slope(y, p) <= 1.1 and 0.9 <= p.sum() / y.sum() <= 1.1
print(f"GATE G-xG (pooled): {'PASS' if gate else 'FAIL'}")
foot = te[te.header == 0]; y = foot.goal.to_numpy(float); print(f"foot shots only: M0 pred/obs {pred['M0 (old)'](foot).sum()/y.sum():.3f}; xG_new pred/obs {pred[f'xG_new ({best})'](foot).sum()/y.sum():.3f}")
# ---------------- (B) empirical xA
Pa["d2"], Pa["ad"], Pa["x"] = Pa.dist ** 2, Pa.angle * Pa.dist, Pa.tx; FA = ["dist", "angle", "x", "d2", "ad", "n_tri", "n3", "n6", "nn", "gk_d", "plen"]
ptr, pte = Pa[~Pa.test].reset_index(drop=True), Pa[Pa.test].reset_index(drop=True)
print(f"\npasses (completed, ground, attacking half, target visible): train {len(ptr)} ({ptr.shot_next.sum()} followed by the receiver's shot), test {len(pte)} ({pte.shot_next.sum()}); test comps: {dict(pte.comp.value_counts())}")
fa = fit(ptr[FA].to_numpy(float), ptr.v.to_numpy(float), clip=1.0, it=300); hyp = lambda d: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(d)), (np.column_stack([d.dist, d.angle, np.zeros(len(d)), d.n_tri, d.gk_d]) - XG0["mu"]) / XG0["sd"]]) @ XG0["b"])))
base = ptr.v.mean(); ok = True
for lab, d in [("POOLED", pte)] + [(c, pte[pte.comp == c]) for c in sorted(pte.comp.unique())]:
    y = d.v.to_numpy(float); p = fa(d[FA].to_numpy(float)); g = ll(y, np.full(len(y), base)) - ll(y, p); lo, hi = mboot(g, d.match.to_numpy()); h = hyp(d)
    print(f"[{lab}] n={len(d)}: observed mean value {y.mean():.4f}; empirical xA mean {p.mean():.4f} (pred/obs {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}, LL gain vs constant {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}]); old hypothetical xA mean {h.mean():.4f} (x{h.mean()/y.mean():.1f} the observed), its slope {slope(y, h):.3f}")
    if lab == "POOLED": ok = lo > 0 and 0.8 <= slope(y, p) <= 1.2 and 0.85 <= p.mean() / y.mean() <= 1.15
print(f"GATE G-xA (pooled test): {'PASS' if ok else 'FAIL'}")
by = pd.cut(pte.dist * 105 / 120, [0, 12, 20, 30, 80]); print("by target distance to goal (m): " + "; ".join(f"{k}: observed {g.v.mean():.3f} | empirical {fa(g[FA].to_numpy(float)).mean():.3f} | old hypothetical {hyp(g).mean():.3f} (n={len(g)})" for k, g in pte.groupby(by, observed=True)))
pd.to_pickle(dict(xG_new=dict(F=FB, **Xnew.coef), xG_geo=dict(F=F2, **Xgeo.coef), xG_M1=dict(F=F1, **X1.coef), xA=dict(F=FA, **fa.coef), best=best), f"{D}/v5_models.pkl")
# ---------------- (C) shot vs pass on test foot shots
def hz(a, end, O):
    R = steps(a, end, O, False, None); lg = np.column_stack([np.ones(len(R)), (R[:, :len(HF)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-lg))).sum()))
def inpoly_tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
fs = te[te.header == 0].reset_index(drop=True); fs["xg_new"] = pred[f"xG_new ({best})"](fs); fs["xg_old"] = pred["M0 (old)"](fs); rows = []
for i, r in fs.iterrows():
    a = np.array([r.x, r.y]); O = np.array(r.opps, float).reshape(-1, 2); gk = np.array(r.gk); ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); bo, bn = 0.0, 0.0
    for p in r.mates:
        p = np.array(p, float); d = float(np.hypot(*((p - a) * SC)))
        if not (3 <= d <= 40) or (p[0] > line and p[0] > 60): continue
        dd = np.hypot(*(O - p).T); dist = float(np.hypot(GX - p[0], 40 - p[1])); an = float(abs(np.arctan2(44 - p[1], GX - p[0]) - np.arctan2(36 - p[1], GX - p[0]))); nt = sum(inpoly_tri(o, p) for o in O); gd = float(np.hypot(GX - gk[0], 40 - gk[1]))
        xa = fa(np.array([[dist, an, p[0], dist ** 2, an * dist, nt, (dd <= 3).sum(), (dd <= 6).sum(), dd.min(), gd, float(np.hypot(*(p - a)))]]))[0]
        h = float(1 / (1 + np.exp(-(np.array([1, *((np.array([dist, an, 0, nt, gd]) - XG0["mu"]) / XG0["sd"])]) @ XG0["b"]))))
        xp = hz(a, p, O); bn = max(bn, xp * xa); bo = max(bo, xp * h)
    rows.append((bn, bo))
fs["epv_new"], fs["epv_old"] = zip(*rows); fs["band"] = pd.cut(fs.dist * 105 / 120, [0, 12, 20, 200], labels=["<12m", "12-20m", ">20m"])
fs["b_new"], fs["b_old"] = fs.epv_new > fs.xg_new, fs.epv_old > fs.xg_old
print(f"\nSHOT vs PASS on test competitions, {len(fs)} foot shots: new (xG_new vs xP x empirical xA): better pass in {fs.b_new.mean()*100:.1f}%, median gap {(fs.epv_new-fs.xg_new).median():+.3f}, sum xG {fs.xg_new.sum():.1f} vs sum best EPV {fs.epv_new.sum():.1f}")
print(f"  old (xG M0 vs xP x hypothetical xA): better pass in {fs.b_old.mean()*100:.1f}%, median gap {(fs.epv_old-fs.xg_old).median():+.3f}, sum xG {fs.xg_old.sum():.1f} vs sum best EPV {fs.epv_old.sum():.1f}")
for c, d in [("by band", None)] + [(b, g) for b, g in fs.groupby("band", observed=True)]:
    if d is not None: print(f"  {c}: n={len(d)}  new {d.b_new.mean()*100:.1f}% (gap>0.05: {((d.epv_new-d.xg_new)>0.05).mean()*100:.1f}%)   old {d.b_old.mean()*100:.1f}%")
for c, d in fs.groupby("comp"): print(f"  {c}: n={len(d)}  new {d.b_new.mean()*100:.1f}%  old {d.b_old.mean()*100:.1f}%")
fs.drop(columns=["opps", "mates"]).to_pickle(f"{D}/v5_svp.pkl")
