"""Stage 2 evidence: is StatsBomb's failed-pass end_location an outcome-independent intended target?"""
import pandas as pd, numpy as np, os
D = os.path.dirname(os.path.abspath(__file__)); I = pd.read_pickle(f"{D}/incomp.pkl")
print(f"incomplete passes with a following opponent event: {len(I)}")
for t, d in I.groupby("ntype"):
    if len(d) >= 200: print(f"  next opponent event {t:<16} n={len(d):>6}: |end_location - opponent event location| median {d.d_next.median():.2f} units, share <= 2 units {(d.d_next<=2).mean()*100:.1f}%")
print(f"  all: median {I.d_next.median():.2f}, share <= 2 units {(I.d_next<=2).mean()*100:.1f}%, <= 5 units {(I.d_next<=5).mean()*100:.1f}%")
