"""Top-down map of the reconstructed frame (own drawing from estimated coordinates; no broadcast content)."""
import sys; sys.path.insert(0, "research/cases"); sys.path.insert(0, "research/dq")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import frame_case as fc
H = fc.homography([np.array(p) for p, _ in fc.LM.values()], [m for _, m in fc.LM.values()])
UW = {k: fc.apply(H, np.array(v)) for k, v in {**fc.PL, **fc.OP}.items()}           # (lateral u, depth w) metres
base = fc.run(); fig, ax = plt.subplots(figsize=(9, 6.6), dpi=160)
ax.add_patch(Rectangle((-34, 0), 68, 40, fc=(0.93, 0.96, 0.92), ec="none"))
L = dict(color="#7a8a7a", lw=1.2)
ax.plot([-34, 34], [0, 0], **L); ax.plot([-20.16, -20.16, 20.16, 20.16], [0, 16.5, 16.5, 0], **L); ax.plot([-9.16, -9.16, 9.16, 9.16], [0, 5.5, 5.5, 0], **L)
ax.add_patch(Rectangle((-3.66, -1.2), 7.32, 1.2, fc="white", ec="#444", lw=1.5)); ax.plot(0, 11, "o", color="#7a8a7a", ms=3)
t = UW["shooter"]; tri = [(t[0], t[1]), (-3.66, 0), (3.66, 0)]; ax.add_patch(Polygon(tri, fc="#f4b183", alpha=0.25, ec="#c55a11", lw=0.8, ls="--"))
for k, (u, w) in UW.items():
    if k == "shooter": ax.scatter(u, w, s=150, c="#c00000", ec="k", zorder=5, marker="*"); ax.annotate("shooter\n(≈8.2 m from goal)", (u, w), (u - 1, w + 2.6), fontsize=8, ha="center", color="#c00000")
    elif k in fc.PL: ax.scatter(u, w, s=80, c="#e06666", ec="k", zorder=4)
    elif k == "GK": ax.scatter(u, w, s=80, c="#2e9e44", ec="k", zorder=4, marker="s"); ax.annotate("GK", (u, w), (u + 0.8, w + 0.6), fontsize=8)
    else: ax.scatter(u, w, s=80, c="#333333", ec="k", zorder=4)
lab = {"R90": "#90", "R27": "#27", "R17": "#17", "R_midleft": "", "R_farleft": "", "R2": "#2"}
for k, nm in lab.items():
    if not nm: continue
    u, w = UW[k]; b = base[k]
    ax.annotate("", (u, w), (t[0], t[1]), arrowprops=dict(arrowstyle="->", color="#1f4e9c", lw=1.6 if k == "R90" else 0.8, alpha=0.9 if k == "R90" else 0.5))
    ax.annotate(f"{nm}: xP {b['xP']:.2f} × xA {b['xA']:.2f}\n= {b['EPV']:.2f}", (u, w), (u + 0.8, w - 3.2), fontsize=8, color="#1f4e9c", weight="bold" if k == "R90" else None)
ax.text(-25.5, 31.2, f"shot xG (our model) {base['xG']:.2f}   |   best pass (#90) xP×xA {base['R90']['EPV']:.2f}\nopponents in shooting triangle: {base['n_tri']} (incl. GK). \nPositions estimated from one broadcast frame (±1-3 m).", fontsize=8, va="top")
ax.set_xlim(-26, 26); ax.set_ylim(-1.8, 32); ax.set_aspect("equal"); ax.set_xlabel("lateral (m, from goal centre)"); ax.set_ylabel("distance from goal line (m)")
ax.set_title("Reconstructed situation at the second shot (frame at 71:48; positions slightly after the kick)", fontsize=10)
ax.scatter([], [], c="#e06666", ec="k", label="attackers"); ax.scatter([], [], c="#333", ec="k", label="defenders"); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("research/cases/frame_case_map.png"); print("saved")
