import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
Cd, Pd, RULES = pd.read_pickle(sys.argv[1])
L = 105.0
Pd["zone"] = pd.cut(Pd.px, [-60, -L / 6, L / 6, 60], labels=["defensive third", "middle third", "attacking third"])
def wilson(k, n):
    p = k / n; z = 1.96; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h
print("population: completed Scenario-A-eligible passes by game:", Pd.groupby("game").size().to_dict(), "total", len(Pd))
print("tracked teammates per pass: min", Pd.n_mates.min(), "max", Pd.n_mates.max())
for name in RULES:
    A = Cd[Cd[name]]
    size = A.groupby("pass").size().reindex(Pd["pass"]).fillna(0).to_numpy(); Pd["size_" + name] = size
    act = Cd[Cd.is_actual].set_index("pass")
    Pd["cov_" + name] = act.loc[Pd["pass"], name].to_numpy()
    print(f"\n== {name} {RULES[name]} ==  candidate-set size: mean {size.mean():.2f}, median {np.median(size):.0f}, IQR {np.percentile(size,25):.0f}-{np.percentile(size,75):.0f}, min {size.min():.0f}; sets with <2 candidates: {(size < 2).sum()}")
    k = int(Pd["cov_" + name].sum()); lo, hi = wilson(k, len(Pd))
    print(f"coverage (actual receiver admissible): {k}/{len(Pd)} = {k / len(Pd):.1%} (95% Wilson {lo:.1%}-{hi:.1%})")
    for col in ("game", "zone", "role"):
        t = Pd.groupby(col, observed=True).agg(n=("pass", "size"), covered=("cov_" + name, "sum")); t["coverage"] = (t.covered / t.n).round(3)
        print(f"  by {col}: " + "; ".join(f"{i}: {int(r.covered)}/{int(r.n)} = {r.coverage:.1%}" for i, r in t.iterrows()))
    ex = act[~act[name]]
    why = []
    for _, r in ex.iterrows():
        lo_, hi_, off = RULES[name]
        why.append("offside" if (r.offside and off) else ("too close (<3 m)" if r.dist < lo_ else ("too far" if r.dist > hi_ else "other")))
    print("  excluded actual receivers by reason:", pd.Series(why).value_counts().to_dict() if why else {})
    if name == "R0_primary":
        print("  EXAMPLES of excluded actual receivers (game, event, passer, dist m, forward m, offside flag):")
        for i, r in ex.head(6).iterrows():
            print(f"    G{int(r.game)} pass {i}: {Pd.set_index('pass').loc[i,'passer']} -> {r.cand}: dist {r.dist:.1f}, forward {r.forward:+.1f}, offside {r.offside}")
        print("  excluded receivers: share forward", f"{(ex.forward > 0).mean():.1%}", "vs covered", f"{(act[act[name]].forward > 0).mean():.1%}", "| median dist excluded", f"{ex.dist.median():.1f}", "vs covered", f"{act[act[name]].dist.median():.1f}")
pd.to_pickle(Pd, sys.argv[2])
