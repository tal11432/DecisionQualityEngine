"""DQ-v10 analysis per PREREG_DQ_v10.txt: passes made when a shot was available."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS as HFT
rng = np.random.default_rng(0); GX = 120.0; SC = np.array([105 / 120, 68 / 80])
Mo = pd.read_pickle(f"{D}/v10_moments.pkl"); M8 = pd.read_pickle(f"{D}/v8_models.pkl"); M6 = pd.read_pickle(f"{D}/v6_models.pkl"); HM = pd.read_pickle(f"{D}/tf_model.pkl")
sgn = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
def tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def hz(a, end, O):
    R = steps(a, end, O, False, None); l = np.column_stack([np.ones(len(R)), (R[:, :len(HFT)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-l))).sum()))
def epv(a, p, O, gk):
    p = np.asarray(p, float); dd = np.hypot(*(O - p).T); dist = float(np.hypot(GX - p[0], 40 - p[1])); an = float(abs(np.arctan2(44 - p[1], GX - p[0]) - np.arctan2(36 - p[1], GX - p[0])))
    f = dict(dist=dist, angle=an, x=p[0], d2=dist ** 2, ad=an * dist, n_tri=sum(tri(o, p) for o in O), n3=int((dd <= 3).sum()), n6=int((dd <= 6).sum()), nn=float(dd.min()), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])),
             plen=float(np.hypot(*(p - a))), fwd=float(p[0] - a[0]), pdist=float(np.hypot(GX - a[0], 40 - a[1])))
    xa = sgn(M6["xA_chain"], [f[k] for k in M6["xA_chain"]["F"]]); xp = hz(a, p, O); return xp * xa, xp, xa
rows = []
for _, r in Mo.iterrows():
    a = np.array([r.x, r.y]); O = np.array(r.opps, float).reshape(-1, 2); gk = np.array(r.gk, float); vs = sgn(M8["Vs"], [r[k] for k in M8["Vs"]["F"]])
    ep, xp, xa = epv(a, r.end, O, gk); ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); best = 0.0
    for p in r.mates:
        p = np.array(p, float)
        if not (3 <= float(np.hypot(*((p - a) * SC))) <= 40) or (p[0] > line and p[0] > 60): continue
        best = max(best, epv(a, p, O, gk)[0])
    rows.append((vs, ep, xp, xa, max(best, ep if False else 0.0)))
Mo["vs"], Mo["e_pass"], Mo["xp"], Mo["xa"], Mo["best"] = zip(*rows); Mo["value"] = np.clip(Mo.v_chain, 0, 0.99); Mo["band"] = pd.cut(Mo.dist * 105 / 120, [0, 12, 20, 30], labels=["<12m", "12-20m", "20-30m"])
Mo.drop(columns=["opps", "mates"]).to_pickle(f"{D}/v10_scored.pkl")
def boot(d, f, n=500):
    um = d.match.unique(); g = {u: np.flatnonzero(d.match.to_numpy() == u) for u in um}; return np.percentile([f(d.iloc[np.concatenate([g[u] for u in rng.choice(um, len(um))])]) for _ in range(n)], [2.5, 97.5])
def report(lab, d, thr):
    c = d[d.vs >= thr]; print(f"\n[{lab}; could-shoot threshold {thr}] ground passes from x>=90: {len(d)}; with a shot worth >= {thr}: {len(c)} ({len(c)/len(d)*100:.1f}%); mean shot value {c.vs.mean():.3f}; mean ex-ante value of the pass taken {c.e_pass.mean():.3f}; best available pass {c.best.mean():.3f}; realised chain {c.value.mean():.3f}")
    if len(c) < 30: return c
    pc, sc = c.e_pass * 0.7 > c.vs, c.e_pass * 1.3 < c.vs; print(f"  was the pass taken right? pass clearly better {pc.mean()*100:.1f}% | shot clearly better {sc.mean()*100:.1f}% | ambiguous {(1-pc.mean()-sc.mean())*100:.1f}%")
    sb = c.best * 1.3 < c.vs; pb = c.best * 0.7 > c.vs; print(f"  versus the BEST pass available: some pass clearly better than the shot {pb.mean()*100:.1f}% | the shot clearly better than every pass {sb.mean()*100:.1f}% | ambiguous {(1-pb.mean()-sb.mean())*100:.1f}%")
    for nm, m in (("completed", c.completed == 1), ("failed", c.completed == 0)):
        d2 = c[m]; print(f"    {nm}: n={len(d2)}, pass clearly better {(d2.e_pass*0.7>d2.vs).mean()*100:.0f}%, shot clearly better {(d2.e_pass*1.3<d2.vs).mean()*100:.0f}%")
    for b, g in c.groupby("band", observed=True): print(f"    {b}: n={len(g)}; pass clearly better {(g.e_pass*0.7>g.vs).mean()*100:.0f}%, shot clearly better {(g.e_pass*1.3<g.vs).mean()*100:.0f}% ; mean shot {g.vs.mean():.3f} vs pass {g.e_pass.mean():.3f}; realised chain {g.value.mean():.3f}")
    s_, p_ = c[c.e_pass * 1.3 < c.vs], c[c.e_pass * 0.7 > c.vs]
    if len(s_) >= 20 and len(p_) >= 20:
        diff = p_.value.mean() - s_.value.mean(); ci = boot(c, lambda x: x[x.e_pass * 0.7 > x.vs].value.mean() - x[x.e_pass * 1.3 < x.vs].value.mean())
        print(f"  KNOWN GROUPS: realised chain of passes classed 'pass clearly better' {p_.value.mean():.4f} (n={len(p_)}) vs 'shot clearly better' {s_.value.mean():.4f} (n={len(s_)}); difference {diff:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  => {'PASS' if ci[0] > 0 else 'FAIL'}")
        print(f"  shot-clearly-better passes: the shot would have been worth {s_.vs.mean():.3f} (model) vs the pass chain realised {s_.value.mean():.3f}; ex-ante pass value {s_.e_pass.mean():.3f}")
    ratio = c.value.mean() / c.e_pass.mean(); ci = boot(c, lambda x: x.value.mean() / x.e_pass.mean()); print(f"  CALIBRATION among could-shoot passes: realised / ex-ante = {ratio:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  => {'within 0.7-1.3' if 0.7 <= ratio <= 1.3 else 'outside 0.7-1.3'}")
    return c
test = Mo[Mo.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023")]; ins = Mo[~Mo.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023")]
print(f"moments: test {len(test)} (comps {dict(test.comp.value_counts())}), in-sample 360 competitions {len(ins)}")
for thr in (0.05, 0.10): report("TEST", test, thr)
report("IN-SAMPLE (reference only)", ins, 0.05)
