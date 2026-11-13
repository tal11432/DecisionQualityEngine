"""One broadcast frame (Hapoel Be'er Sheva vs Hapoel Tel Aviv, 71:48, user-supplied screenshot) -> pitch coordinates -> xG / xP / xA.
Pixel coordinates are read by eye from the screenshot (2000-px-wide scale) and carry error; a Monte-Carlo over those errors is reported.
Models: research/dq/stage1b_models.pkl (xG) and tf_model.pkl (xP hazard, NOT recalibrated: no Israeli data exist). Outcome of the real action is not used."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research/dq"); from tfcommon import steps, FEATS as HF
XG = pd.read_pickle("research/dq/stage1b_models.pkl")["xG"]; HM = pd.read_pickle("research/dq/tf_model.pkl"); GX = 120.0
ang = lambda x, y: float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def intri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def xg(pos, opps, gk, extra=0):
    x = np.array([[np.hypot(GX - pos[0], 40 - pos[1]), ang(*pos), 0.0, sum(intri(o, pos) for o in opps) + extra, np.hypot(GX - gk[0], 40 - gk[1])]])
    return float(1 / (1 + np.exp(-(np.column_stack([np.ones(1), (x - XG["mu"]) / XG["sd"]]) @ XG["b"])))[0])
def xp(a, end, O):
    R = steps(a, end, O, False, None); lg = np.column_stack([np.ones(len(R)), (R[:, :len(HF)] - HM["mu"]) / HM["sd"]]) @ HM["b"]
    return float(np.exp(np.log1p(-1 / (1 + np.exp(-lg))).sum()))
def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst): A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    return np.linalg.svd(np.array(A, float))[2][-1].reshape(3, 3)
def apply(H, p): q = H @ np.array([p[0], p[1], 1.0]); return q[:2] / q[2]
# landmarks (pixel -> (u lateral m, w depth from goal line m)); 6-yard box half width 9.16 m, depth 5.5 m; goal posts at +-3.66 m
LM = {"6yd left side x goal line": ((909, 390), (-9.16, 0)), "6yd left front corner": ((685, 426), (-9.16, 5.5)),
      "6yd right front corner": ((1300, 614), (9.16, 5.5)), "left post base": ((1087, 434), (-3.66, 0))}
CHECK = ((1190, 910), (20.16, 16.5))        # 18-yard-box front-right corner: out-of-sample check
PL = {"shooter": (765, 450), "R90": (914, 524), "R27": (1017, 623), "R17": (475, 555), "R_midleft": (395, 442), "R_farleft": (222, 385), "R2": (310, 768)}
OP = {"B_near_shooter": (673, 487), "B_grounded": (865, 437), "GK": (973, 448), "B_post1": (1100, 473), "B_post2": (1152, 499), "B_farleft1": (530, 398), "B_farleft2": (275, 512), "B_farleft3": (185, 405)}
def to_sb(H, p):
    u, w = apply(H, p); return np.array([GX - w * 120 / 105, 40 + u * 80 / 68]), (u, w)
def run(jit=0.0, rng=None, verbose=False):
    J = (lambda: rng.normal(0, jit, 2)) if rng is not None else (lambda: np.zeros(2))
    H = homography([np.array(p) + J() for p, _ in LM.values()], [m for _, m in LM.values()])
    pos = {k: to_sb(H, np.array(v) + J()) for k, v in {**PL, **OP}.items()}
    if verbose:
        print("check, 18-yard corner predicted vs true (u,w): %s vs %s" % (np.round(apply(H, CHECK[0]), 1), CHECK[1]))
        for k, (s, (u, w)) in pos.items(): print(f"  {k:<15} lateral {u:+6.1f} m  from goal line {w:5.1f} m")
    sh = pos["shooter"][0]; opps = [pos[k][0] for k in OP]; gk = pos["GK"][0]; O = np.array(opps)
    out = {"xG": xg(sh, opps, gk), "xG_+1def": xg(sh, opps, gk, extra=1), "shot_dist_m": float(np.hypot(*(np.array([GX, 40]) - sh) * np.array([105 / 120, 68 / 80]))), "n_tri": sum(intri(o, sh) for o in opps)}
    line = max(sorted(o[0] for o in opps)[-2], sh[0])
    for k in PL:
        if k == "shooter": continue
        p = pos[k][0]; d = float(np.hypot(*((p - sh) * np.array([105 / 120, 68 / 80]))))
        ok = 3 <= d <= 40 and not (p[0] > line and p[0] > 60); a, P_ = xg(p, opps, gk), xp(sh, p, O)
        out[k] = dict(d=d, admissible=ok, xA=a, xP=P_, EPV=P_ * a, from_goal=float(np.hypot(*((np.array([GX, 40]) - p) * np.array([105 / 120, 68 / 80])))))
    return out
if __name__ == "__main__":
    base = run(verbose=True); print(f"\nBASE: shot {base['shot_dist_m']:.1f} m from goal centre, opponents in shooting triangle {base['n_tri']}, xG {base['xG']:.3f} (+1 defender {base['xG_+1def']:.3f})")
    for k in PL:
        if k != "shooter": b = base[k]; print(f"  {k:<10} d {b['d']:5.1f} m, from goal {b['from_goal']:5.1f} m, admissible {b['admissible']}, xP {b['xP']:.2f}, xA {b['xA']:.3f}, EPV {b['EPV']:.3f}")
    rng = np.random.default_rng(0); R = [run(10.0, rng) for _ in range(500)]
    print("\nMONTE CARLO (500 draws; every landmark and player pixel jittered N(0, 10 px)):")
    q = lambda v: f"median {np.median(v):.3f} [5%-95% {np.percentile(v,5):.3f}, {np.percentile(v,95):.3f}]"
    print("  xG shot:", q([r["xG"] for r in R]))
    for k in ("R90", "R27", "R17"): print(f"  {k}: xP {q([r[k]['xP'] for r in R])}; xA {q([r[k]['xA'] for r in R])}; EPV {q([r[k]['EPV'] for r in R])}")
    best = [max((r[k]["EPV"] for k in PL if k != "shooter" and r[k]["admissible"]), default=0) for r in R]
    print("  best EPV among admissible teammates:", q(best), "; share of draws with best EPV > xG(shot): %.0f%%" % (100 * np.mean([b > r["xG"] for b, r in zip(best, R)])))
    print("  share of draws with R90 EPV > shot xG: %.0f%%; with R27: %.0f%%" % tuple(100 * np.mean([r[k]["EPV"] > r["xG"] for r in R]) for k in ("R90", "R27")))
