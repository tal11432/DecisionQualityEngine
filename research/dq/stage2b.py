"""Stage 2 with 360 frames: target identification gate, then Ps calibration (EU20 train -> WC22 test)."""
import numpy as np, pandas as pd, os, importlib.util
D = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
spec = importlib.util.spec_from_file_location("s", f"{D}/stage1b.py")
P = pd.read_pickle(f"{D}/passes360.pkl")
print(f"in-scope open-play ground passes: {len(P)} (EU20 {sum(P.comp=='EU20')}, WC22 {sum(P.comp=='WC22')}); complete {P.complete.mean()*100:.1f}%")
for comp, d in [*P.groupby("comp"), ("ALL", P)]:
    c, i = d[d.complete == 1], d[d.complete == 0]
    print(f"[{comp}] 360 frame present: complete {c.has360.mean()*100:.1f}% vs incomplete {i.has360.mean()*100:.1f}%")
    c, i = c[c.has360], i[i.has360]
    print(f"      identified: complete {c.ident.mean()*100:.1f}% (n={len(c)}) vs incomplete {i.ident.mean()*100:.1f}% (n={len(i)}); no teammate in cone: {c.no_cone.mean()*100:.1f}% vs {i.no_cone.mean()*100:.1f}%; "
          f"mean visible teammates {c.n_mates.mean():.1f} vs {i.n_mates.mean():.1f}")
    print(f"      completed: identified teammate == teammate nearest end_location in {c[c.ident==1].near_end_is_ident.mean()*100:.1f}% of identified")
h = P[P.has360]; c, i = h[h.complete == 1], h[h.complete == 0]
g_i = abs(c.ident.mean() - i.ident.mean()) <= 0.10; g_a = c[c.ident == 1].near_end_is_ident.mean() >= 0.80; g_f = abs(P[P.complete == 1].has360.mean() - P[P.complete == 0].has360.mean()) <= 0.10
print(f"\nGATE G2-target: (i) coverage gap {'PASS' if g_i else 'FAIL'}; (ii) agreement {'PASS' if g_a else 'FAIL'}; (iii) frame availability {'PASS' if g_f else 'FAIL'}  => {'PASS' if (g_i and g_a and g_f) else 'FAIL'}")
