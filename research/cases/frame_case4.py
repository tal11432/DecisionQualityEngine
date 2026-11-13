"""Frame 3 (71:47, just before the FIRST shot, official xG 0.166). Pixel readings on the 2000-px scale, measured on a zoomed crop.
Official shot location from the Redwood map: goal line y=422 px, penalty-area front y=170 (16.5 m), 6-yard front y=337 (5.5 m); centre x=543.5; 14.0 px/m laterally, 15.3 px/m in depth;
highlighted circle (653, 315)."""
import sys; sys.path.insert(0, "research/cases"); sys.path.insert(0, "research/dq")
import numpy as np, frame_case as fc
fc.LM = {"6yd left side x goal line": ((1032, 401), (-9.16, 0)), "6yd left front corner": ((830, 426), (-9.16, 5.5)),
         "6yd right front corner": ((1395, 605), (9.16, 5.5)), "left post base": ((1196, 443), (-3.66, 0))}
fc.PL = {"shooter": (740, 430), "R90": (908, 551), "R27": (1004, 636), "R17": (601, 548), "R_a": (345, 398), "R_b": (405, 370), "R2": (248, 765)}
fc.OP = {"B_near_shooter": (721, 478), "B_diving": (800, 456), "GK": (1126, 445), "B_post1": (1040, 478), "B_post2": (1122, 528), "B_far1": (560, 395), "B_far2": (378, 375), "B_far3": (300, 485), "B_far4": (190, 400), "B_far5": (200, 730)}
u_map = (653 - 543.5) / 14.0; w_map = (422 - 315) / 15.3
if __name__ == "__main__":
    print(f"official location (map): lateral {u_map:.1f} m, {w_map:.1f} m from goal line, {np.hypot(u_map, w_map):.1f} m from goal centre")
    b0 = fc.run(verbose=True)
    print(f"\nA) shooter at the frame-based position: {b0['shot_dist_m']:.1f} m from goal, triangle opponents {b0['n_tri']}, xG {b0['xG']:.3f}")
    rng = np.random.default_rng(0); q = lambda v: f"median {np.median(v):.3f} [5%-95% {np.percentile(v,5):.3f}, {np.percentile(v,95):.3f}]"
    for lab, uw in (("B) official location, camera side (lateral -7.8)", (-u_map, w_map)), ("C) official location, mirrored (+7.8)", (u_map, w_map))):
        fc.SHOOTER_UW = uw; b = fc.run(); R = []
        for _ in range(300): fc.SHOOTER_UW = (uw[0] + rng.normal(0, 0.5), uw[1] + rng.normal(0, 0.5)); R.append(fc.run(10.0, rng))
        print(f"\n{lab}: {b['shot_dist_m']:.1f} m from goal, triangle opponents {b['n_tri']}, xG {b['xG']:.3f} (league 0.166); Monte Carlo xG {q([r['xG'] for r in R])}; share >= 0.166: {100*np.mean([r['xG']>=0.166 for r in R]):.0f}%")
        for k in ("R90", "R27", "R17"):
            print(f"  {k}: d {b[k]['d']:.1f} m, from goal {b[k]['from_goal']:.1f} m, admissible {b[k]['admissible']}, xP {b[k]['xP']:.2f}, xA {b[k]['xA']:.3f}, EPV {b[k]['EPV']:.3f}; share of draws EPV > shot xG {100*np.mean([r[k]['EPV']>r['xG'] for r in R]):.0f}%; EPV {q([r[k]['EPV'] for r in R])}")
        bestd = [max((r[k]['EPV'] for k in fc.PL if k != 'shooter' and r[k]['admissible']), default=0) > r['xG'] for r in R]; print(f"  best admissible EPV > shot xG in {100*np.mean(bestd):.0f}% of draws")
    fc.SHOOTER_UW = None
