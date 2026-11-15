"""DQ-v11 analysis per PREREG_DQ_v11.txt: one numeric value per decision."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS as HFT
rng = np.random.default_rng(0); GX = 120.0; SC = np.array([105 / 120, 68 / 80])
Mo = pd.read_pickle(f"{D}/v11_moments.pkl"); Sh = pd.read_pickle(f"{D}/v11_shots.pkl"); M8 = pd.read_pickle(f"{D}/v8_models.pkl"); M6 = pd.read_pickle(f"{D}/v6_models.pkl"); HM = pd.read_pickle(f"{D}/tf_model.pkl")
sgn = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
def tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def hz(a, end, O):
    if float(np.hypot(*(np.asarray(end, float) - a))) < 1e-6: return 1.0     # zero-length pass: nothing to intercept
    R = steps(a, end, O, False, None); l = np.column_stack([np.ones(len(R)), (R[:, :len(HFT)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-l))).sum()))
def epv(a, p, O, gk):
    p = np.asarray(p, float); dd = np.hypot(*(O - p).T); dist = float(np.hypot(GX - p[0], 40 - p[1])); an = float(abs(np.arctan2(44 - p[1], GX - p[0]) - np.arctan2(36 - p[1], GX - p[0])))
    f = dict(dist=dist, angle=an, x=p[0], d2=dist ** 2, ad=an * dist, n_tri=sum(tri(o, p) for o in O), n3=int((dd <= 3).sum()), n6=int((dd <= 6).sum()), nn=float(dd.min()), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])),
             plen=float(np.hypot(*(p - a))), fwd=float(p[0] - a[0]), pdist=float(np.hypot(GX - a[0], 40 - a[1])))
    xa = sgn(M6["xA_chain"], [f[k] for k in M6["xA_chain"]["F"]]); xp = hz(a, p, O); return xp * xa, xp, xa

SS = (0.7, 1.0, 1.3); TESTRE = "Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024"
def best_alt(a, O, gk, mates):
    ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); best = 0.0
    for p in mates:
        p = np.array(p, float)
        if not (3 <= float(np.hypot(*((p - a) * SC))) <= 40) or (p[0] > line and p[0] > 60): continue
        best = max(best, epv(a, p, O, gk)[0])
    return best
rows = []
for kind, df in (("shot", Sh), ("pass", Mo)):
    for _, r in df.iterrows():
        a = np.array([r.x, r.y]); O = np.array(r.opps, float).reshape(-1, 2); gk = np.array(r.gk, float); vs = sgn(M8["Vs"], [r[k] for k in M8["Vs"]["F"]]); bst = best_alt(a, O, gk, r.mates)
        ech = vs if kind == "shot" else epv(a, r.end, O, gk)[0]
        reg, eff = [], []
        for sc in SS:
            vc = vs if kind == "shot" else sc * ech; vb = max(vs, sc * bst, vc); reg.append(vb - vc); eff.append(vc / vb if vb > 0 else 1.0)
        rows.append(dict(kind=kind, comp=r.comp, match=r.match, period=r.period, clock=f"{int(r.minute):02d}:{int(r.second):02d}", team=r.team, event_index=r["index"], outcome=r.outcome, dist_m=r.dist * 105 / 120, completed=(r.completed if kind == "pass" else np.nan),
                         v_chosen=vs if kind == "shot" else ech, shot_value=vs, best_pass_value=bst, regret=reg[1], regret_low=min(reg), regret_high=max(reg), efficiency=eff[1], reference_xg=(r.sb_xg + r.cont if kind == "shot" else np.nan),
                         realised=(np.clip(r.v_chain, 0, 0.99) if kind == "pass" else np.nan), best_alt=("shot" if vs >= bst else "pass") if kind == "pass" else ("pass" if bst > vs else "shot"), match_key=r.match))
R = pd.DataFrame(rows); R["test"] = R.comp.str.contains(TESTRE)
R["verdict"] = np.where(R.regret_low >= 0.01, "clearly suboptimal", np.where(R.regret_high < 0.01, "near-optimal", "ambiguous")); R["band"] = pd.cut(R.dist_m, [0, 12, 20, 40], labels=["<12m", "12-20m", ">20m"])
cols = ["comp", "match", "period", "clock", "team", "event_index", "kind", "outcome", "dist_m", "v_chosen", "shot_value", "best_pass_value", "best_alt", "regret", "regret_low", "regret_high", "efficiency", "verdict"]
R[R.test][cols].round(4).to_csv(f"{D}/decision_values_test.csv", index=False); R.to_pickle(f"{D}/v11_scored.pkl")
def summ(lab, d):
    print(f"\n[{lab}] decisions {len(d)}: shots {(d.kind=='shot').sum()}, passes {(d.kind=='pass').sum()}")
    for k, g in d.groupby("kind"):
        v = g.verdict.value_counts(normalize=True) * 100; print(f"  {k}: mean V_chosen {g.v_chosen.mean():.4f}; mean regret {g.regret.mean():.4f} (median {g.regret.median():.4f}); mean efficiency {g.efficiency.mean():.3f}; near-optimal {v.get('near-optimal',0):.1f}% | ambiguous {v.get('ambiguous',0):.1f}% | clearly suboptimal {v.get('clearly suboptimal',0):.1f}%")
        for b, g2 in g.groupby("band", observed=True): vv = g2.verdict.value_counts(normalize=True) * 100; print(f"     {b}: n={len(g2)}, mean regret {g2.regret.mean():.4f}, near-optimal {vv.get('near-optimal',0):.0f}% / ambiguous {vv.get('ambiguous',0):.0f}% / clearly suboptimal {vv.get('clearly suboptimal',0):.0f}%")
summ("TEST", R[R.test]); summ("IN-SAMPLE (reference)", R[~R.test])
T = R[R.test]; P = T[(T.kind == "pass")]; no, cs = P[P.verdict == "near-optimal"], P[P.verdict == "clearly suboptimal"]
def bd(d): 
    um = d.match.unique(); g = {u: np.flatnonzero(d.match.to_numpy() == u) for u in um}; f = lambda x: x[x.verdict == "near-optimal"].realised.mean() - x[x.verdict == "clearly suboptimal"].realised.mean()
    return np.percentile([f(d.iloc[np.concatenate([g[u] for u in rng.choice(um, len(um))])]) for _ in range(500)], [2.5, 97.5])
ci = bd(P); print(f"\nCHECK 1 known groups (passes, test): realised chain of near-optimal {no.realised.mean():.4f} (n={len(no)}) vs clearly suboptimal {cs.realised.mean():.4f} (n={len(cs)}); difference {no.realised.mean()-cs.realised.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] => {'PASS' if ci[0] > 0 else 'FAIL'}")
pc = P[P.completed == 1]; print(f"CHECK 2 completed passes: realised / ex-ante = {pc.realised.mean()/pc.v_chosen.mean():.3f} => {'PASS' if 0.8 <= pc.realised.mean()/pc.v_chosen.mean() <= 1.2 else 'FAIL'} (all passes {P.realised.mean()/P.v_chosen.mean():.3f}; failed {P[P.completed==0].realised.mean()/P[P.completed==0].v_chosen.mean():.3f})")
S_ = T[T.kind == "shot"]; print(f"CHECK 3 shots: mean V_chosen {S_.v_chosen.mean():.4f} vs reference StatsBomb xG + continuation {S_.reference_xg.mean():.4f} (ratio {S_.v_chosen.mean()/S_.reference_xg.mean():.3f}) => {'PASS' if 0.85 <= S_.v_chosen.mean()/S_.reference_xg.mean() <= 1.15 else 'FAIL'}")
print("\nEXAMPLES, largest regret (test; no player names):")
print(T.sort_values("regret", ascending=False).head(10)[["comp", "match", "period", "clock", "team", "kind", "outcome", "v_chosen", "shot_value", "best_pass_value", "regret", "regret_low", "regret_high", "verdict"]].round(3).to_string(index=False))
print("\nEXAMPLES, near-optimal and a high-value choice:")
print(T[T.verdict == "near-optimal"].sort_values("v_chosen", ascending=False).head(5)[["comp", "match", "period", "clock", "team", "kind", "outcome", "v_chosen", "shot_value", "best_pass_value", "regret"]].round(3).to_string(index=False))
