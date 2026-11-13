"""Israeli frames with the DQ-v6 models: shot value = xG_geo + continuation; pass value = xP x xA_chain (possession-chain value within 10 s). Same positions as v5_cases."""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "research/cases"); sys.path.insert(0, "research/dq")
import frame_case as fc
M5 = pd.read_pickle("research/dq/v5_models.pkl"); M6 = pd.read_pickle("research/dq/v6_models.pkl"); GX = 120.0
sg = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
def feats_xg(a, opps, gk):
    O = np.array(opps, float).reshape(-1, 2); dd = np.hypot(*(O - a).T); v = np.array([GX, 40.0]) - a; dist = float(np.hypot(*v)); an = fc.ang(*a)
    return [dist, an, 0.0, sum(fc.intri(o, a) for o in opps), float(np.hypot(GX - gk[0], 40 - gk[1])), abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9), int((dd <= 2).sum()), float(dd.min()), dist * an, np.log(dist)]
def shot_val(a, opps, gk):
    f = feats_xg(a, opps, gk); return sg(M5["xG_geo"], f) + sg(M6["cont"], f[:5])
fc.XG_FN = shot_val
def xa(p, opps, gk, a):
    O = np.array(opps, float).reshape(-1, 2); dd = np.hypot(*(O - p).T); dist = float(np.hypot(GX - p[0], 40 - p[1])); an = fc.ang(*p)
    return sg(M6["xA_chain"], [dist, an, p[0], dist ** 2, an * dist, sum(fc.intri(o, p) for o in opps), int((dd <= 3).sum()), int((dd <= 6).sum()), float(dd.min()), float(np.hypot(GX - gk[0], 40 - gk[1])), float(np.hypot(*(p - a))), p[0] - a[0], float(np.hypot(GX - a[0], 40 - a[1]))])
fc.XA_FN = xa
def case(lab, mod, uw, league):
    import importlib; importlib.import_module(mod); fc.SHOOTER_UW = uw; b = fc.run(); rng = np.random.default_rng(0); R = []
    for _ in range(300): fc.SHOOTER_UW = (uw[0] + rng.normal(0, 0.5), uw[1] + rng.normal(0, 0.5)); R.append(fc.run(10.0, rng))
    q = lambda v: f"median {np.median(v):.3f} [5%-95% {np.percentile(v,5):.3f}, {np.percentile(v,95):.3f}]"
    print(f"\n{lab}: shot value (xG_geo + continuation) {b['xG']:.3f} (official xG {league}); Monte Carlo {q([r['xG'] for r in R])}")
    for k in ("R90", "R27", "R17"): print(f"  {k}: xP {b[k]['xP']:.2f}, chain xA {b[k]['xA']:.4f}, pass value {b[k]['EPV']:.4f}; share of draws pass > shot {100*np.mean([r[k]['EPV'] > r['xG'] for r in R]):.0f}%")
    print(f"  best admissible pass > shot in {100*np.mean([max((r[k]['EPV'] for k in fc.PL if k != 'shooter' and r[k]['admissible']), default=0) > r['xG'] for r in R]):.0f}% of draws"); fc.SHOOTER_UW = None
case("Second shot (71:48)", "frame_case2", (-3.68, 6.4), 0.378)
import frame_case4 as f4; fc.LM, fc.PL, fc.OP = f4.fc.LM, f4.fc.PL, f4.fc.OP
case("First shot (71:47)", "frame_case4", (-f4.u_map, f4.w_map), 0.166)
