"""DQ-v12b: floor calibration with data-based noise (PREREG_DQ_v12b.txt). Usage: python eval_v12b.py SIGMA_POS SIGMA_PASS SIGMA_SHOT. Needs v11_shots.pkl (fetch_v11.py)."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS as HFT
rng = np.random.default_rng(0); GX = 120.0; SC = np.array([105 / 120, 68 / 80]); M8 = pd.read_pickle(f"{D}/v8_models.pkl"); M6 = pd.read_pickle(f"{D}/v6_models.pkl"); HM = pd.read_pickle(f"{D}/tf_model.pkl")
SIG_POS, SIG_PASS, SIG_SHOT = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); print(f'noise: position sd {SIG_POS} m, pass scale sd {SIG_PASS}, shot scale sd {SIG_SHOT}')
# ---------- (A) noise criterion on shots
Sh = pd.read_pickle(f"{D}/v11_shots.pkl"); Sh = Sh[Sh.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024")].reset_index(drop=True)
sgn = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
ang = lambda x, y: float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def hz(a, end, O):
    if float(np.hypot(*(end - a))) < 1e-6: return 1.0
    Rr = steps(a, end, O, False, None); l = np.column_stack([np.ones(len(Rr)), (Rr[:, :len(HFT)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-l))).sum()))
def values(a, O, Tm, gk):
    dd = np.hypot(*(O - a).T); v = np.array([GX, 40.0]) - a; dist = float(np.hypot(*v)); an = ang(*a)
    c = dict(x=a[0], y=a[1], dist=dist, angle=an, n_tri=sum(tri(o, a) for o in O), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), gk_lat=float(abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9)), n_opp2=int((dd <= 2).sum()), nn_opp=float(dd.min()), da=dist * an, ldist=float(np.log(dist)))
    vs = sgn(M8["Vs"], [c[k] for k in M8["Vs"]["F"]]); ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); best = 0.0
    for p in Tm:
        if not (3 <= float(np.hypot(*((p - a) * SC))) <= 40) or (p[0] > line and p[0] > 60): continue
        dp = np.hypot(*(O - p).T); dist_p = float(np.hypot(GX - p[0], 40 - p[1])); an_p = ang(*p)
        f = dict(dist=dist_p, angle=an_p, x=p[0], d2=dist_p ** 2, ad=an_p * dist_p, n_tri=sum(tri(o, p) for o in O), n3=int((dp <= 3).sum()), n6=int((dp <= 6).sum()), nn=float(dp.min()), gk_d=c["gk_d"], plen=float(np.hypot(*(p - a))), fwd=float(p[0] - a[0]), pdist=dist)
        best = max(best, hz(a, p, O) * sgn(M6["xA_chain"], [f[k] for k in M6["xA_chain"]["F"]]))
    return vs, best
def cls(vs, bp): return "pass" if 0.7 * bp > vs else ("shot" if 1.3 * bp < vs else "none")
def margin(vs, bp, k): return (0.7 * bp - vs) if k == "pass" else (vs - 1.3 * bp)
sample = Sh.iloc[rng.choice(len(Sh), 300, replace=False)]; J = np.array([120 / 105, 80 / 68]); rows = []
for _, r in sample.iterrows():
    a0 = np.array([r.x, r.y], float); O0 = np.array(r.opps, float).reshape(-1, 2); T0 = np.array(r.mates, float).reshape(-1, 2); g0 = np.array(r.gk, float)
    if len(T0) == 0: continue
    vs0, bp0 = values(a0, O0, T0, g0); k0 = cls(vs0, bp0)
    if k0 == "none": continue
    same = 0; n = 150
    for _ in range(n):
        a = a0 + rng.normal(0, SIG_POS, 2) * J; O = O0 + rng.normal(0, SIG_POS, O0.shape) * J; Tm = T0 + rng.normal(0, SIG_POS, T0.shape) * J; gk = g0 + rng.normal(0, SIG_POS, 2) * J
        vs, bp = values(a, O, Tm, gk); vs *= np.exp(rng.normal(0, SIG_SHOT)); bp *= np.exp(rng.normal(0, SIG_PASS)); same += (cls(vs, bp) == k0)
    rows.append((k0, margin(vs0, bp0, k0), same / n, vs0, bp0))
S = pd.DataFrame(rows, columns=["cls", "c", "stab", "vs", "bp"]); edges = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 9]
print(f"\n(A) NOISE CRITERION: {len(S)} shots with a clear verdict at FLOOR=0 (pass {int((S.cls=='pass').sum())}, shot {int((S.cls=='shot').sum())}); overall mean stability {S.stab.mean():.3f}")
bins = []
for lo, hi in zip(edges[:-1], edges[1:]):
    g = S[(S.c >= lo) & (S.c < hi)]; bins.append((lo, g.stab.mean() if len(g) else np.nan, len(g))); print(f"   margin c in [{lo}, {'inf' if hi==9 else hi}): n={len(g):>3}, mean stability {g.stab.mean() if len(g) else float('nan'):.3f}, share >= 0.9: {(g.stab>=0.9).mean()*100 if len(g) else float('nan'):.0f}%")
floor = None
for i, (lo, m, n) in enumerate(bins):
    if n and m >= 0.90 and all((mm >= 0.90) for (_, mm, nn) in bins[i:] if nn): floor = lo; break
print(f"\nFLOOR* = {floor if floor is not None else '0.05 (not reached: no bin with mean stability >= 0.90 that persists)'}")
by = S.assign(lvl=pd.cut(S.vs.where(S.cls == 'shot', S.bp), [0, 0.05, 0.1, 0.3, 2])); print("stability by the larger of the two values:", "; ".join(f"{k}: {g.stab.mean():.2f} (n={len(g)})" for k, g in by.groupby("lvl", observed=True)))
