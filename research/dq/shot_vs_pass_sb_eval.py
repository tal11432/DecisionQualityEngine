"""Analysis for PREREG_SHOT_SB.txt."""
import os, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); Sh, Cn, excl = pd.read_pickle(f"{D}/shot_vs_pass_sb.pkl"); rng = np.random.default_rng(0)
Cn["EPV"], Cn["EPV1"] = Cn.xP * Cn.xA, Cn.xP * Cn.xA1
def best(col, sub=None):
    c = Cn if sub is None else Cn[sub]; b = c.loc[c.groupby("sid")[col].idxmax()].set_index("sid")
    m = Sh.set_index("sid").join(b[[col, "xP", "xA", "d_m"]]); m[col] = m[col].fillna(0.0); m["gap"] = m[col] - m.xg; m["better"] = m.gap > 0; return m
def ci(m):
    um = m.match.unique(); g = {u: np.flatnonzero(m.match.to_numpy() == u) for u in um}; v = m.better.to_numpy()
    bs = [v[np.concatenate([g[u] for u in rng.choice(um, len(um))])].mean() for _ in range(1000)]; return np.percentile(bs, [2.5, 97.5])
m = best("EPV"); lo, hi = ci(m)
print(f"open-play foot shots {len(Sh)} (excluded: no goalkeeper in frame {excl['no_gk']}); goals {Sh.goal.sum()}; xG sum {Sh.xg.sum():.1f} (StatsBomb xG {Sh.sb_xg.sum():.1f})")
print(f"visible teammates per shot median {Sh.n_mates.median():.0f}, admissible candidates per shot median {Cn.groupby('sid').size().reindex(Sh.sid, fill_value=0).median():.0f}; shots with no candidate {(Cn.groupby('sid').size().reindex(Sh.sid, fill_value=0)==0).sum()}")
print(f"\nPRIMARY: shots with a pass option of higher EPV: {m.better.sum()} of {len(m)} = {m.better.mean()*100:.1f}% [95% CI {lo*100:.1f}, {hi*100:.1f}]; median gap {m.gap.median():+.3f}; sum xG {m.xg.sum():.1f} vs sum best EPV {m.EPV.sum():.1f}")
print("gap distribution: " + ", ".join(f"> {t}: {(m.gap > t).mean()*100:.1f}%" for t in (0, 0.02, 0.05, 0.10)))
for c, d in m.groupby("comp"): l, h = ci(d); print(f"  {c}: n={len(d)} better {d.better.mean()*100:.1f}% [{l*100:.1f}, {h*100:.1f}], median gap {d.gap.median():+.3f}")
m["band"] = pd.cut(m.dist_m, [0, 12, 20, 200], labels=["<12m", "12-20m", ">20m"])
for b, d in m.groupby("band", observed=True):
    l, h = ci(d); print(f"  {b}: n={len(d)} better {d.better.mean()*100:.1f}% [{l*100:.1f}, {h*100:.1f}]; median shot xG {d.xg.median():.3f}; median best EPV {d.EPV.median():.3f}; gap>0.05 {(d.gap>0.05).mean()*100:.1f}%")
print(f"best option: median xP {m.xP.median():.2f}, median xA {m.xA.median():.3f}, median distance {m.d_m.median():.1f} m")
print("\nSENSITIVITY")
for lab, mm in (("+1 defender in receiver triangle", best("EPV1")), ("candidates 3-25 m", best("EPV", Cn.d_m <= 25))):
    print(f"  {lab}: better {mm.better.mean()*100:.1f}%, median gap {mm.gap.median():+.3f}, sum best {mm.iloc[:, list(mm.columns).index('EPV1' if 'EPV1' in mm else 'EPV')].sum():.1f}; by band " +
          ", ".join(f"{b} {mm[pd.cut(mm.dist_m,[0,12,20,200],labels=['<12m','12-20m','>20m'])==b].better.mean()*100:.0f}%" for b in ("<12m", "12-20m", ">20m")))
print("\nCHECK (outcome used only here): goals vs xG by group")
for lab, d in (("better pass existed", m[m.better]), ("no better pass", m[~m.better])): print(f"  {lab}: n={len(d)}, goals {d.goal.sum()} vs xG {d.xg.sum():.1f} (ratio {d.goal.sum()/d.xg.sum():.2f}); StatsBomb xG {d.sb_xg.sum():.1f}")
m.to_pickle(f"{D}/shot_vs_pass_sb_m.pkl")
