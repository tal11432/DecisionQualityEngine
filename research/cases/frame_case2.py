"""Frame 2: the moment just BEFORE the kick (ball visible at the shooter's foot). Same method as frame_case.py; pixel readings (2000-px scale) re-measured.
Shot origin = ball position."""
import sys; sys.path.insert(0, "research/cases"); sys.path.insert(0, "research/dq")
import numpy as np, frame_case as fc
fc.LM = {"6yd left side x goal line": ((907, 395), (-9.16, 0)), "6yd left front corner": ((674, 425), (-9.16, 5.5)),
         "6yd right front corner": ((1291, 612), (9.16, 5.5)), "left post base": ((1079, 437), (-3.66, 0))}
fc.PL = {"shooter": (773, 435), "R90": (914, 524), "R27": (1010, 630), "R17": (475, 555), "R_midleft": (390, 438), "R_farleft": (230, 383), "R2": (300, 768)}
fc.OP = {"B_near_shooter": (627, 485), "B_grounded": (869, 443), "GK": (999, 437), "B_post1": (1033, 476), "B_post2": (1100, 504), "B_farleft1": (530, 395), "B_farleft2": (265, 512), "B_farleft3": (192, 402)}
if __name__ == "__main__":
    base = fc.run(verbose=True)
    print(f"\nBASE (frame 2): shot {base['shot_dist_m']:.1f} m from goal centre, opponents in shooting triangle {base['n_tri']}, xG {base['xG']:.3f} (+1 defender {base['xG_+1def']:.3f})")
    for k in fc.PL:
        if k != "shooter": b = base[k]; print(f"  {k:<10} d {b['d']:5.1f} m, from goal {b['from_goal']:5.1f} m, admissible {b['admissible']}, xP {b['xP']:.2f}, xA {b['xA']:.3f}, EPV {b['EPV']:.3f}")
    rng = np.random.default_rng(0); R = [fc.run(10.0, rng) for _ in range(500)]
    q = lambda v: f"median {np.median(v):.3f} [5%-95% {np.percentile(v,5):.3f}, {np.percentile(v,95):.3f}]"
    print("\nMONTE CARLO (500 draws, N(0,10 px) on every landmark and player):\n  xG shot:", q([r["xG"] for r in R]), "; share of draws with xG >= 0.378 (league value): %.0f%%" % (100 * np.mean([r["xG"] >= 0.378 for r in R])))
    for k in ("R90", "R27", "R17"): print(f"  {k}: xP {q([r[k]['xP'] for r in R])}; xA {q([r[k]['xA'] for r in R])}; EPV {q([r[k]['EPV'] for r in R])}")
    print("  share of draws with R90 EPV > shot xG: %.0f%%; with R27: %.0f%%" % tuple(100 * np.mean([r[k]["EPV"] > r["xG"] for r in R]) for k in ("R90", "R27")))
    # what if the league's xG (0.378) is right: would #90 still beat it? scale xA by the same factor as the shot
    f = 0.378 / base["xG"]; print(f"\nIf our xG is too low by the factor that would match the league ({f:.2f}x) and xA is too low by the same factor: R90 EPV {base['R90']['EPV']*f:.3f} vs 0.378 -> ratio unchanged ({base['R90']['EPV']/base['xG']:.2f}x)")
