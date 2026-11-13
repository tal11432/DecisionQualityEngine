"""Analysis for PREREG_SHOT_SB2.txt."""
import os, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); Sh, Cn, excl = pd.read_pickle(f"{D}/shot_vs_pass_sb2.pkl"); rng = np.random.default_rng(0)
W = {"W Euro 2025", "W Euro 2022", "WWC 2023", "WWC 2019", "FA WSL 2023/24", "Liga F 2023/24", "Frauen Bundesliga 2023/24", "Serie A Women 2023/24", "NWSL 2023"}
Sh["women"] = Sh.comp.isin(W); Cn["EPV"], Cn["EPV1"] = Cn.xP * Cn.xA, Cn.xP * Cn.xA1
def best(col):
    b = Cn.loc[Cn.groupby("sid")[col].idxmax()].set_index("sid"); m = Sh.set_index("sid").join(b[[col, "xP", "xA"]]); m[col] = m[col].fillna(0.0); m["gap"] = m[col] - m.xg; m["better"] = m.gap > 0
    m["band"] = pd.cut(m.dist_m, [0, 12, 20, 200], labels=["<12m", "12-20m", ">20m"]); return m
m, m1 = best("EPV"), best("EPV1")
def slope(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); Z = np.column_stack([np.ones(len(lp)), lp]); b = np.array([0.0, 1.0])
    for _ in range(100):
        q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (y - q)); b += s
        if np.abs(s).max() < 1e-10: break
    return b[1]
def ci(d, col="better"):
    um = d.match.unique(); g = {u: np.flatnonzero(d.match.to_numpy() == u) for u in um}; v = d[col].to_numpy(float); bs = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(500)]
    return np.percentile(bs, [2.5, 97.5])
def block(lbl, d, d1):
    lo, hi = ci(d); bands = {b: d[d.band == b].better.mean() * 100 for b in ("<12m", "12-20m", ">20m")}
    red = 1 - d1.gap.median() / d.gap.median() if d.gap.median() > 0 else np.nan
    print(f"{lbl:<26} shots {len(d):>5} goals {int(d.goal.sum()):>4} xG {d.xg.sum():>6.1f} (O/E {d.goal.sum()/d.xg.sum():.2f}, slope {slope(d.goal.to_numpy(), d.xg.to_numpy()):.2f}) | better {d.better.mean()*100:4.1f}% [{lo*100:.1f},{hi*100:.1f}] | <12 {bands['<12m']:.0f}% 12-20 {bands['12-20m']:.0f}% >20 {bands['>20m']:.0f}% | median gap {d.gap.median():+.3f} (+1 def: {d1.gap.median():+.3f}, cut {red*100:.0f}%)")
    return bands, red
print(f"excluded (no goalkeeper in frame): {excl['no_gk']}; shots with no admissible candidate: {int((~m.index.isin(Cn.sid)).sum())}")
for grp, mask in (("MEN", ~m.women), ("WOMEN", m.women)):
    print(f"\n===== {grp} ====="); d, d1 = m[mask], m1[mask]; allb, allred = block("pooled", d, d1)
    ok_order = []; reds = []
    for c in sorted(d.comp.unique()):
        dc = d[d.comp == c]
        if len(dc) < 150: print(f"  {c:<24} shots {len(dc)} (<150, not assessed individually)"); continue
        b, r = block("  " + c, dc, d1[d1.comp == c]); ok_order.append(b["<12m"] < b["12-20m"] < b[">20m"]); reds.append(r)
    if grp == "MEN":
        p = d.copy(); slp = slope(p.goal.to_numpy(), p.xg.to_numpy()); oe = p.goal.sum() / p.xg.sum(); base = {"<12m": 43.5, "12-20m": 72.1, ">20m": 90.3}
        R1 = 0.8 <= slp <= 1.2 and 0.85 <= 1 / oe * 1 <= 1.15 if False else (0.8 <= slp <= 1.2 and 0.85 <= p.xg.sum() / p.goal.sum() <= 1.15)
        R2 = all(ok_order) and allb["<12m"] < allb["12-20m"] < allb[">20m"]; R3 = all(abs(allb[b] - base[b]) <= 7 for b in base) and abs(d.better.mean() * 100 - 72.9) <= 5
        R4 = all(0.3 <= r <= 0.7 for r in reds)
        print(f"\nREPLICATION (men): R1 calibration (slope {slp:.2f}, mean xG/observed {p.xg.sum()/p.goal.sum():.2f}) {'PASS' if R1 else 'FAIL'}; R2 ordering {'PASS' if R2 else 'FAIL'}; R3 shares within tolerance {'PASS' if R3 else 'FAIL'}; R4 defender sensitivity {'PASS' if R4 else 'FAIL'} (cuts {min(reds)*100:.0f}-{max(reds)*100:.0f}%)")
print("\nMost recent (men, 2024): " + ", ".join(f"{c}: {m[m.comp==c].better.mean()*100:.1f}% of {len(m[m.comp==c])}" for c in ("Euro 2024", "Copa America 2024")))
m.to_pickle(f"{D}/shot_vs_pass_sb2_m.pkl")
