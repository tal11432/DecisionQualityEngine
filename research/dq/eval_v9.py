"""DQ-v9 analysis per PREREG_DQ_v9.txt."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS as HFT
rng = np.random.default_rng(0); GX = 120.0; SC = np.array([105 / 120, 68 / 80])
B = pd.read_pickle(f"{D}/v9_base.pkl"); P = pd.read_pickle(f"{D}/v9_p360.pkl"); S = pd.read_pickle(f"{D}/v9_shots.pkl"); M5 = pd.read_pickle(f"{D}/v5_models.pkl"); M6 = pd.read_pickle(f"{D}/v6_models.pkl"); HM = pd.read_pickle(f"{D}/tf_model.pkl")
def fit(X, y, lam=1.0, it=200, clip=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1]); R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    b[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b)); s = np.clip(np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (y - p) - R @ b), -clip, clip); b += s
        if np.abs(s).max() < 1e-8: break
    f = lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b))); f.coef = dict(mu=mu, sd=sd, b=b); return f
ll = lambda y, p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(1 - np.clip(p, 1e-9, 1 - 1e-9)))
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
def slope(y, p):
    Z = np.column_stack([np.ones(len(p)), lg(p)]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def auc(y, p): r = pd.Series(p).rank().to_numpy(); n1 = (y > 0.5).sum(); return (r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
def mboot(v, mt):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(500)]; return np.percentile(b, [2.5, 97.5])
# ----- baseline from all matches (cross-fitted by competition)
def bfeat(d):
    dist = np.hypot(GX - d.tx, 40 - d.ty); an = np.abs(np.arctan2(44 - d.ty, GX - d.tx) - np.arctan2(36 - d.ty, GX - d.tx)); return pd.DataFrame(dict(tx=d.tx, dist=dist, angle=an, d2=dist ** 2, ad=an * dist, plen=d.plen, fwd=d.tx - d.sx, pdist=np.hypot(GX - d.sx, 40 - d.sy), counter=d.counter, setp=d.setp, nprev=np.minimum(d.nprev, 15), tsince=np.minimum(d.tsince, 60)))
BF = list(bfeat(B.iloc[:2]).columns); Bt, Bs = B[~B.test].reset_index(drop=True), B[B.test]; Xb = bfeat(Bt).to_numpy(float); yb = np.clip(Bt.chain.to_numpy(float), 0, 0.99)
print(f"baseline data: {len(Bt)} train passes from {Bt.comp.nunique()} competitions ({(Bt.chain>0).mean()*100:.1f}% followed by a chain shot)")
bcomps = sorted(Bt.comp.unique()); base_cf = {}; base_all = fit(Xb, yb)
for c in bcomps:
    m = (Bt.comp == c).to_numpy(); base_cf[c] = fit(Xb[~m], yb[~m])
def base_pred(d, comp_wise):
    X = bfeat(d.assign(ty=d.ty if "ty" in d else 40.0)).to_numpy(float) if False else None
# p360 lacks sx/sy/ty: build baseline features from what p360 has
def bfeat360(d): return pd.DataFrame(dict(tx=d.tx, dist=d.dist, angle=d.angle, d2=d.dist ** 2, ad=d.angle * d.dist, plen=d.plen, fwd=d.fwd, pdist=d.pdist, counter=d.counter, setp=d.setp, nprev=np.minimum(d.nprev, 15), tsince=np.minimum(d.tsince, 60)))
P["x"], P["d2"], P["ad"] = P.tx, P.dist ** 2, P.angle * P.dist; P["nprev"], P["tsince"] = np.minimum(P.nprev, 15), np.minimum(P.tsince, 60)
X360 = bfeat360(P).to_numpy(float); bl = np.empty(len(P))
for c in P.comp.unique():
    m = (P.comp == c).to_numpy(); bl[m] = (base_cf[c] if c in base_cf else base_all)(X360[m])
P["base"] = lg(bl)
FA6 = ["dist", "angle", "x", "d2", "ad", "n_tri", "n3", "n6", "nn", "gk_d", "plen", "fwd", "pdist"]; NEW = ["counter", "setp", "nprev", "tsince", "att_box", "def_box", "sup10", "ahead25", "opp_between", "opp5_passer"]
F1, F3 = FA6 + NEW, FA6 + NEW + ["base"]; P["v"] = np.clip(P.chain, 0, 0.99); P["has"] = (P.nsh > 0).astype(float)
tr, te = P[~P.test].reset_index(drop=True), P[P.test].reset_index(drop=True)
print(f"360 passes: train {len(tr)} ({tr.has.sum():.0f} with a chain shot), test {len(te)} ({te.has.sum():.0f}); mean chain value train {tr.v.mean():.4f}, test {te.v.mean():.4f}")
def mk(kind, d):
    if kind == "V0": f = fit(d[FA6].to_numpy(float), d.v.to_numpy(float)); return lambda x: f(x[FA6].to_numpy(float)), None
    if kind == "V1": f = fit(d[F1].to_numpy(float), d.v.to_numpy(float)); return lambda x: f(x[F1].to_numpy(float)), None
    F = F1 if kind == "V2" else F3; s = d[d.has > 0.5]; pm = fit(d[F].to_numpy(float), d.has.to_numpy(float)); em = fit(s[F].to_numpy(float), s.v.to_numpy(float))
    return (lambda x: pm(x[F].to_numpy(float)) * em(x[F].to_numpy(float))), (lambda x: pm(x[F].to_numpy(float)))
comps = [c for c in sorted(tr.comp.unique()) if (tr.comp == c).sum() >= 150]; loco = {}
for kind in ("V0", "V1", "V2", "V3"):
    tot = 0; n = 0
    for c in comps:
        a, b = tr[tr.comp != c], tr[tr.comp == c]; f, _ = mk(kind, a); tot += ll(b.v.to_numpy(float), f(b)).sum(); n += len(b)
    loco[kind] = tot / n; print(f"  LOCO log loss {kind}: {loco[kind]:.5f}")
best = min(("V1", "V2", "V3"), key=lambda k: loco[k]); print(f"selected: {best}")
f0, _ = mk("V0", tr); fb, fbp = mk(best, tr); gate_ok = True; slopes = []
for lab, d in [("POOLED", te)] + [(c, te[te.comp == c]) for c in sorted(te.comp.unique()) if (te.comp == c).sum() >= 150]:
    y = d.v.to_numpy(float); p0, pb = f0(d), fb(d); g = ll(y, p0) - ll(y, pb); lo, hi = mboot(g, d.match.to_numpy()); sl = slope(y, pb)
    a0 = auc(d.has.to_numpy(), p0); ab = auc(d.has.to_numpy(), fbp(d) if fbp else pb)
    print(f"[{lab}] n={len(d)}: observed {y.mean():.4f}; V0 predicted {p0.mean():.4f} (ratio {p0.mean()/y.mean():.3f}, slope {slope(y, p0):.3f}); {best} predicted {pb.mean():.4f} (ratio {pb.mean()/y.mean():.3f}, slope {sl:.3f}); LL gain {best} vs V0 {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}]; AUC(chain shot) V0 {a0:.3f} -> {best} {ab:.3f}")
    if lab == "POOLED": gate_ok = lo > 0 and 0.9 <= pb.mean() / y.mean() <= 1.1 and 0.8 <= sl <= 1.2
    else: slopes.append(sl)
gate_ok = gate_ok and all(0.7 <= s <= 1.3 for s in slopes); print(f"GATE G-V9: {'PASS' if gate_ok else 'FAIL'} (per-competition slopes {[round(float(s), 2) for s in slopes]})")
# receipt-frame diagnostic
R = P.dropna(subset=["nn_rec"]); Rtr, Rte = R[~R.test], R[R.test]; FR = F1 + ["nn_rec", "n3_rec", "n6_rec"]
fa = fit(Rtr[F1].to_numpy(float), Rtr.v.to_numpy(float)); fr = fit(Rtr[FR].to_numpy(float), Rtr.v.to_numpy(float)); y = Rte.v.to_numpy(float); g = ll(y, fa(Rte[F1].to_numpy(float))) - ll(y, fr(Rte[FR].to_numpy(float))); lo, hi = mboot(g, Rte.match.to_numpy())
print(f"\nDIAGNOSTIC (not deployable): knowing the defenders at RECEPTION adds LL gain {-g.mean():+.5f} (V1 -> V1+receipt frame; subset n={len(Rte)}), CI [{-hi:+.5f},{-lo:+.5f}]; AUC {auc(Rte.has.to_numpy(), fa(Rte[F1].to_numpy(float))):.3f} -> {auc(Rte.has.to_numpy(), fr(Rte[FR].to_numpy(float))):.3f}")
# ----- shot vs pass with the selected model
def hz(a, end, O):
    Rr = steps(a, end, O, False, None); l = np.column_stack([np.ones(len(Rr)), (Rr[:, :len(HFT)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-l))).sum()))
def tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
sgn = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
inb = lambda p: (p[0] >= 102) and (18 <= p[1] <= 62)
ts = S[S.test & (S.header == 0)].reset_index(drop=True); geo = M5["xG_geo"]; ts["xg"] = [sgn(geo, r[geo["F"]].to_numpy(float)) for _, r in ts.iterrows()]; ts["cont_hat"] = [sgn(M6["cont"], r[M6["cont"]["F"]].to_numpy(float)) for _, r in ts.iterrows()]; rows = []
for _, r in ts.iterrows():
    a = np.array([r.x, r.y]); O = np.array(r.opps, float).reshape(-1, 2); Tm = np.array(r.mates, float).reshape(-1, 2); gk = np.array(r.gk); ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); b0 = bn = 0.0; ok_ = False
    opp5 = int((np.hypot(*(O - a).T) <= 5).sum()); att_box = int(sum(inb(q) for q in Tm)); def_box = int(sum(inb(q) for q in O))
    for p in Tm:
        d = float(np.hypot(*((p - a) * SC)))
        if not (3 <= d <= 40) or (p[0] > line and p[0] > 60): continue
        ok_ = True; dd = np.hypot(*(O - p).T); tm = np.hypot(*(Tm - p).T); dist = float(np.hypot(GX - p[0], 40 - p[1])); an = float(abs(np.arctan2(44 - p[1], GX - p[0]) - np.arctan2(36 - p[1], GX - p[0]))); nt = sum(tri(o, p) for o in O); gd = float(np.hypot(GX - gk[0], 40 - gk[1]))
        row = dict(dist=dist, angle=an, x=p[0], d2=dist ** 2, ad=an * dist, n_tri=nt, n3=(dd <= 3).sum(), n6=(dd <= 6).sum(), nn=dd.min(), gk_d=gd, plen=float(np.hypot(*(p - a))), fwd=p[0] - a[0], pdist=float(np.hypot(GX - a[0], 40 - a[1])),
                   counter=int(r.pattern == "counter"), setp=int(r.pattern == "set"), nprev=min(r.nprev, 15), tsince=min(r.tsince, 60), att_box=att_box, def_box=def_box, sup10=int((tm <= 10).sum()), ahead25=int(((Tm[:, 0] > p[0]) & (tm <= 25)).sum()),
                   opp_between=int(((O[:, 0] > p[0]) & (np.abs(O[:, 1] - p[1]) <= 12)).sum()), opp5_passer=opp5)
        bx = bfeat360(pd.DataFrame([dict(tx=p[0], dist=dist, angle=an, plen=row["plen"], fwd=row["fwd"], pdist=row["pdist"], counter=row["counter"], setp=row["setp"], nprev=r.nprev, tsince=r.tsince)])).to_numpy(float); row["base"] = float(lg(base_all(bx))[0])
        df1 = pd.DataFrame([row]); xp = hz(a, p, O); b0 = max(b0, xp * f0(df1)[0]); bn = max(bn, xp * fb(df1)[0])
    rows.append((b0, bn, ok_))
ts["p_old"], ts["p_new"], ts["ok"] = zip(*rows); ts = ts[ts.ok].reset_index(drop=True); ts["shot_val"] = ts.xg + ts.cont_hat
def cls(d, c, lo=0.7, hi=1.3): return (d[c] * lo > d.shot_val), (d[c] * hi < d.shot_val)
print(f"\nSHOT vs PASS on {len(ts)} test foot shots (shot value = xG_geo + continuation):")
for c, nm in (("p_old", "V0 (DQ-v6/v7)"), ("p_new", f"{best} (new)")):
    pc, sc = cls(ts, c); print(f"  {nm:<16} better pass {(ts[c] > ts.shot_val).mean()*100:.1f}%; THREE-WAY (x0.7..x1.3): pass clearly better {pc.mean()*100:.1f}%, shot clearly better {sc.mean()*100:.1f}%, ambiguous {(1-pc.mean()-sc.mean())*100:.1f}%; "
          f"curve " + ", ".join(f"s={s}: {(ts[c]*s > ts.shot_val).mean()*100:.0f}%" for s in (0.7, 0.85, 1.0, 1.15, 1.3)))
ts["band"] = pd.cut(ts.dist * 105 / 120, [0, 12, 20, 200], labels=["<12m", "12-20m", ">20m"])
for b, g in ts.groupby("band", observed=True): print(f"    {b}: n={len(g)}  V0 better {(g.p_old>g.shot_val).mean()*100:.0f}%  new better {(g.p_new>g.shot_val).mean()*100:.0f}%  new three-way {cls(g,'p_new')[0].mean()*100:.0f}/{cls(g,'p_new')[1].mean()*100:.0f}/{(1-cls(g,'p_new')[0].mean()-cls(g,'p_new')[1].mean())*100:.0f}")
pd.to_pickle(dict(selected=best, F=(F1 if best != "V3" else F3), **{f"V0": f0 and None}), f"{D}/v9_meta.pkl")
