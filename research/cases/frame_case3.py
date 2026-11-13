"""Official shot location (Redwood map screenshot) vs reconstruction from the broadcast frame. Map calibration: goal line y=413 px, penalty-area front y=163 (16.5 m),
6-yard front y=328 (5.5 m); penalty-area width 262-825 px (40.32 m), 6-yard 413-673 px (18.32 m); centre x=543.5. Highlighted circle centre (595, 316)."""
import sys; sys.path.insert(0, "research/cases"); sys.path.insert(0, "research/dq")
import numpy as np, frame_case as fc, frame_case2   # frame_case2 sets the pre-kick frame's players/landmarks
u_map = (595 - 543.5) / 14.0; w_map = (413 - 316) / 15.2
print(f"official location (from the map): lateral {u_map:.1f} m from goal centre, {w_map:.1f} m from goal line, {np.hypot(u_map, w_map):.1f} m from goal centre (uncertainty about +-0.5 m)")
rec = fc.run(); print(f"reconstruction from the broadcast frame: lateral -7.0 m, 4.4 m from goal line, {rec['shot_dist_m']:.1f} m from goal centre; xG {rec['xG']:.3f}")
rng = np.random.default_rng(0)
for lab, uw in (("official location, same side as the camera view (lateral -3.7)", (-u_map, w_map)), ("official location, mirrored (lateral +3.7)", (u_map, w_map))):
    fc.SHOOTER_UW = uw; b = fc.run()
    R = []
    for _ in range(300):
        fc.SHOOTER_UW = (uw[0] + rng.normal(0, 0.5), uw[1] + rng.normal(0, 0.5)); R.append(fc.run(10.0, rng))
    q = lambda v: f"median {np.median(v):.3f} [5%-95% {np.percentile(v,5):.3f}, {np.percentile(v,95):.3f}]"
    print(f"\n{lab}:\n  shot {b['shot_dist_m']:.1f} m from goal, triangle opponents {b['n_tri']}, xG {b['xG']:.3f}  | with position +-0.5 m and players +-10 px: xG {q([r['xG'] for r in R])}; share >= 0.378: {100*np.mean([r['xG']>=0.378 for r in R]):.0f}%")
    for k in ("R90", "R27"): print(f"  {k}: xP {b[k]['xP']:.2f}, xA {b[k]['xA']:.3f}, EPV {b[k]['EPV']:.3f}; share of draws with EPV > shot xG: {100*np.mean([r[k]['EPV']>r['xG'] for r in R]):.0f}%")
fc.SHOOTER_UW = None
