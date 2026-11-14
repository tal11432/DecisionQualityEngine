"""Decision-moment models (DQ-v8) applied to the two Israeli shooting contexts: P(shoot | context), value of a shot Vs, value of a typical pass Vp."""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "research/cases"); sys.path.insert(0, "research/dq")
import frame_case as fc
M = pd.read_pickle("research/dq/v8_models.pkl"); GX = 120.0
sgm = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
def feats(a, opps, gk):
    O = np.array(opps, float).reshape(-1, 2); dd = np.hypot(*(O - a).T); v = np.array([GX, 40.0]) - a; dist = float(np.hypot(*v)); an = fc.ang(*a)
    d = dict(dist=dist, angle=an, n_tri=sum(fc.intri(o, a) for o in opps), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), gk_lat=abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9), n_opp2=int((dd <= 2).sum()), nn_opp=float(dd.min()), da=dist * an, ldist=float(np.log(dist)), x=a[0])
    return [d[k] for k in M["C"]["F"]]
def case(lab, mod, uw):
    import importlib; importlib.import_module(mod); rng = np.random.default_rng(0); R = []
    for i in range(301):
        J = (lambda: rng.normal(0, 10.0, 2)) if i else (lambda: np.zeros(2)); u = (uw[0] + (rng.normal(0, .5) if i else 0), uw[1] + (rng.normal(0, .5) if i else 0))
        H = fc.homography([np.array(p) + J() for p, _ in fc.LM.values()], [m for _, m in fc.LM.values()])
        pos = {k: fc.to_sb(H, np.array(v) + J())[0] for k, v in {**fc.PL, **fc.OP}.items()}; sh = np.array([GX - u[1] * 120 / 105, 40 + u[0] * 80 / 68]); opps = [pos[k] for k in fc.OP]; f = feats(sh, opps, pos["GK"])
        R.append((sgm(M["C"], f), sgm(M["Vs"], f), sgm(M["Vp"], f)))
    R = np.array(R); q = lambda v: f"{v[0]:.3f} (MC 5-95%: {np.percentile(v[1:],5):.3f}-{np.percentile(v[1:],95):.3f})"
    print(f"\n{lab}: P(a typical player shoots from here) {q(R[:,0])}; value of a shot {q(R[:,1])}; value of a typical pass {q(R[:,2])}; share of draws with shot value > typical pass {100*np.mean(R[1:,1] > R[1:,2]):.0f}%")
case("Second shot (71:48)", "frame_case2", (-3.68, 6.4))
import frame_case4 as f4; fc.LM, fc.PL, fc.OP = f4.fc.LM, f4.fc.PL, f4.fc.OP
case("First shot (71:47)", "frame_case4", (-f4.u_map, f4.w_map))
