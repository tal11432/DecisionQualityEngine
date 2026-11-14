"""DQ-v8 analysis per PREREG_DQ_v8.txt."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
Sh = pd.read_pickle(f"{D}/v8_shots.pkl"); Mo = pd.read_pickle(f"{D}/v8_moments.pkl"); XG0 = pd.read_pickle(f"{D}/stage1b_models.pkl")["xG"]
def fit(X, y, lam=1.0, it=300, clip=None):
    mu, sd = X.mean(0), X.std(0) + 1e-12; Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1]); R = lam * np.eye(Z.shape[1]); R[0, 0] = 0
    b[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (y - p) - R @ b)
        if clip: s = np.clip(s, -clip, clip)
        b += s
        if np.abs(s).max() < 1e-9: break
    f = lambda Xn: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b))); f.coef = dict(mu=mu, sd=sd, b=b); return f
ll = lambda y, p: -(y * np.log(np.clip(p, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(1 - np.clip(p, 1e-9, 1 - 1e-9)))
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
def slope(y, p):
    Z = np.column_stack([np.ones(len(p)), lg(p)]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def platt(y, p):
    Z = np.column_stack([np.ones(len(p)), lg(p)]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return lambda p2: 1 / (1 + np.exp(-(b[0] + b[1] * lg(p2)))), b
def auc(y, p): r = pd.Series(p).rank().to_numpy(); n1 = (y > 0.5).sum(); return (r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
def mboot(v, mt):
    um = np.unique(mt); g = {u: np.flatnonzero(mt == u) for u in um}; b = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(500)]; return np.percentile(b, [2.5, 97.5])
F1 = ["dist", "angle", "header", "n_tri", "gk_d"]; F2 = F1 + ["gk_lat", "n_opp2", "nn_opp", "da", "ldist"]; F3 = F2 + ["first_time", "volley", "pat_counter", "pat_set", "prior_cross", "prior_through", "prior_cutback"]
HK = {"hd10": ("dist", 10), "hd16": ("dist", 16), "hd24": ("dist", 24), "hd34": ("dist", 34), "ha15": ("angle", 0.15), "ha30": ("angle", 0.3), "ha60": ("angle", 0.6)}
for df in (Sh,):
    for k, (c, kn) in HK.items(): df[k] = np.maximum(0, df[c] - kn)
    df["i_lat"], df["i_tri"], df["i_ft"] = df.gk_lat * df.dist, df.n_tri * df.dist, df.first_time * df.dist
HF = list(HK); FS1 = F3 + HF; FS2 = FS1 + ["i_lat", "i_tri", "i_ft"]; FG1 = F2 + HF
tr, te = Sh[~Sh.test].reset_index(drop=True), Sh[Sh.test].reset_index(drop=True)
print(f"shots: train {len(tr)} ({tr.goal.sum()} goals, {tr.comp.nunique()} competitions), test {len(te)} ({te.goal.sum()} goals): " + ", ".join(f"{c} {n}" for c, n in te.comp.value_counts().items()))
M = {"M2": F2, "M3": F3, "S1": FS1, "S2": FS2, "G1": FG1}; comps = [c for c in sorted(tr.comp.unique()) if (tr.comp == c).sum() >= 30]; loco = {}
for nm, F in M.items():
    P = np.full(len(tr), np.nan)
    for c in comps:
        a, b = (tr.comp != c).to_numpy(), (tr.comp == c).to_numpy(); P[b] = fit(tr.loc[a, F].to_numpy(float), tr.goal[a].to_numpy(float))(tr.loc[b, F].to_numpy(float))
    ok = ~np.isnan(P); y = tr.goal.to_numpy(float)[ok]; f_pl, b = platt(y, P[ok]); loco[nm] = (ll(y, P[ok]).mean(), ll(y, f_pl(P[ok])).mean(), f_pl, b)
    print(f"  LOCO log loss {nm}: raw {loco[nm][0]:.5f}, with Platt {loco[nm][1]:.5f} (a={b[0]:+.3f}, b={b[1]:.3f})")
choice = min([(loco[n][i], n, i) for n in ("S1", "S2") for i in (0, 1)]); _, bn, bi = choice; print(f"selected: {bn} {'with Platt recalibration' if bi else 'raw'}")
def build(nm, F, use_platt):
    f = fit(tr[F].to_numpy(float), tr.goal.to_numpy(float)); g = loco[nm][2] if use_platt else (lambda p: p); return (lambda d: g(f(d[F].to_numpy(float)))), f
new_fn, new_f = build(bn, M[bn], bi); geo_fn, geo_f = build("G1", FG1, bi and True)
pred = {"M0 (old)": lambda d: 1 / (1 + np.exp(-(np.column_stack([np.ones(len(d)), (d[F1].to_numpy(float) - XG0["mu"]) / XG0["sd"]]) @ XG0["b"]))), "M3 (DQ-v5 xG_new)": lambda d: fit(tr[F3].to_numpy(float), tr.goal.to_numpy(float))(d[F3].to_numpy(float)),
        f"xG_new2 ({bn}{'+Platt' if bi else ''})": new_fn, f"xG_geo2 (G1{'+Platt' if bi else ''})": geo_fn, "StatsBomb xG": lambda d: d.sb_xg.to_numpy(float)}
gate = False
for lab, d in [("POOLED", te)] + [(c, te[te.comp == c]) for c in sorted(te.comp.unique()) if (te.comp == c).sum() >= 150]:
    y = d.goal.to_numpy(float); p0 = pred["M0 (old)"](d); p3 = pred["M3 (DQ-v5 xG_new)"](d); print(f"[{lab}] n={len(d)} goals={int(y.sum())}")
    for nm, fn in pred.items():
        p = fn(d); g0 = ll(y, p0) - ll(y, p); g3 = ll(y, p3) - ll(y, p); lo, hi = mboot(g0, d.match.to_numpy()); lo3, hi3 = mboot(g3, d.match.to_numpy())
        print(f"   {nm:<26} pred/obs {p.sum()/y.sum():.3f} slope {slope(y, p):.3f} AUC {auc(y, p):.3f} LL {ll(y, p).mean():.4f} | gain vs M0 {g0.mean():+.5f} [{lo:+.5f},{hi:+.5f}] | vs M3 {g3.mean():+.5f} [{lo3:+.5f},{hi3:+.5f}]")
        if lab == "POOLED" and nm.startswith("xG_new2"): gate = lo > 0 and 0.9 <= slope(y, p) <= 1.1 and 0.9 <= p.sum() / y.sum() <= 1.1 and hi3 >= 0 and g3.mean() >= -0.0005
print(f"GATE G-xG2 (pooled): {'PASS' if gate else 'FAIL'}")
# ---------------- (8) decision moments
Sm = Sh[(Sh.has360) & (Sh.x >= 90) & (Sh.header == 0)].copy(); Sm["value"] = np.clip(Sm.sb_xg + Sm.cont, 0, 0.99); Sm["shot"] = 1
Mo["value"] = np.clip(Mo.v_chain, 0, 0.99); Mo["shot"] = 0; CF = ["dist", "angle", "n_tri", "gk_d", "gk_lat", "n_opp2", "nn_opp", "da", "ldist", "x"]
A = pd.concat([Sm[CF + ["comp", "match", "test", "value", "shot"]], Mo[CF + ["comp", "match", "test", "value", "shot"]]]).reset_index(drop=True); Atr, Ate = A[~A.test], A[A.test]
print(f"\ndecision moments: shots {int(A.shot.sum())} (train {int(Atr.shot.sum())}, test {int(Ate.shot.sum())}), passes {int((1-A.shot).sum())} (train {int((1-Atr.shot).sum())}, test {int((1-Ate.shot).sum())}); shot share {A.shot.mean()*100:.1f}%")
C = fit(Atr[CF].to_numpy(float), Atr.shot.to_numpy(float)); y = Ate.shot.to_numpy(float); p = C(Ate[CF].to_numpy(float)); g = ll(y, np.full(len(y), Atr.shot.mean())) - ll(y, p); lo, hi = mboot(g, Ate.match.to_numpy())
okC = 0.8 <= slope(y, p) <= 1.2 and 0.9 <= p.mean() / y.mean() <= 1.1
print(f"C = P(shoot | context): test AUC {auc(y, p):.3f}, slope {slope(y, p):.3f}, predicted shot share {p.mean()*100:.1f}% vs observed {y.mean()*100:.1f}%, LL gain {g.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  => {'PASS' if okC else 'FAIL'}")
by = pd.cut(Ate.dist * 105 / 120, [0, 8, 12, 16, 20, 30]); print("   shot share by distance (m): " + "; ".join(f"{k}: observed {g2.shot.mean()*100:.0f}% | C {C(g2[CF].to_numpy(float)).mean()*100:.0f}% (n={len(g2)})" for k, g2 in Ate.groupby(by, observed=True)))
ok = True; V = {}
for nm, sh in (("Vs (shot)", 1), ("Vp (typical pass)", 0)):
    d_tr, d_te = Atr[Atr.shot == sh], Ate[Ate.shot == sh]; f = fit(d_tr[CF].to_numpy(float), d_tr.value.to_numpy(float), clip=1.0); V[sh] = f; y = d_te.value.to_numpy(float); p = f(d_te[CF].to_numpy(float))
    g = ll(y, np.full(len(y), d_tr.value.mean())) - ll(y, p); lo, hi = mboot(g, d_te.match.to_numpy()); o = 0.85 <= p.mean() / y.mean() <= 1.15 and 0.8 <= slope(y, p) <= 1.2; ok &= o
    print(f"{nm}: test n={len(y)}, observed mean {y.mean():.4f}, predicted {p.mean():.4f} (ratio {p.mean()/y.mean():.3f}, slope {slope(y, p):.3f}, LL gain {g.mean():+.5f} [{lo:+.5f},{hi:+.5f}])  => {'PASS' if o else 'FAIL'}")
print(f"GATE G-moment: {'PASS' if (okC and ok) else 'FAIL'}")
Ate = Ate.assign(vs=V[1](Ate[CF].to_numpy(float)), vp=V[0](Ate[CF].to_numpy(float)), c=C(Ate[CF].to_numpy(float)))
s_ = Ate[Ate.shot == 1]; p_ = Ate[Ate.shot == 0]
print(f"\nACTUAL SHOTS (test, n={len(s_)}): value of a shot from the context {s_.vs.mean():.3f} vs typical pass {s_.vp.mean():.3f}; share with Vs > Vp {np.mean(s_.vs > s_.vp)*100:.1f}%; realised mean value of the shots taken {s_.value.mean():.3f}")
q = p_[p_.vs >= 0.05]; um = q.match.unique(); gg = {u: np.flatnonzero(q.match.to_numpy() == u) for u in um}; d_ = (q.vs - q.value).to_numpy(); bs = [d_[np.concatenate([gg[u] for u in rng.choice(um, len(um))])].mean() for _ in range(500)]
print(f"ACTUAL PASSES from contexts where a shot would have been worth >= 0.05 (test, n={len(q)}): mean shot value {q.vs.mean():.3f} vs mean realised chain of the pass taken {q.value.mean():.3f} (difference {d_.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]); share where typical pass < shot value {np.mean(q.vp < q.vs)*100:.1f}%")
for b, g2 in Ate.groupby(by, observed=True): print(f"   {b}: moments {len(g2)}; mean Vs {g2.vs.mean():.3f}, Vp {g2.vp.mean():.3f}; share of moments where the player shot {g2.shot.mean()*100:.0f}%; share of contexts with Vs > Vp {np.mean(g2.vs > g2.vp)*100:.0f}%")
pd.to_pickle(dict(xG_new2=dict(F=M[bn], platt=(loco[bn][3] if bi else None), **new_f.coef), xG_geo2=dict(F=FG1, platt=(loco["G1"][3] if bi else None), **geo_f.coef), C=dict(F=CF, **C.coef), Vs=dict(F=CF, **V[1].coef), Vp=dict(F=CF, **V[0].coef), sel=bn), f"{D}/v8_models.pkl")
