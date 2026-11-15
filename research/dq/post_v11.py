"""POST-HOC step after eval_v11.py (not pre-registered): the pre-registered ABSOLUTE regret verdict failed its known-groups check because an absolute threshold mixes the quality of the situation with the quality of the decision.
This script adds a scale-free (relative) verdict from the efficiency at the three pass-value scales and rewrites the CSV with both verdicts."""
import numpy as np, pandas as pd
D = "research/dq/"; R = pd.read_pickle(D + "v11_scored.pkl"); rng = np.random.default_rng(0)
def effs(d):
    out = []
    for s in (0.7, 1.0, 1.3):
        vc = np.where(d.kind == "shot", d.shot_value, s * d.v_chosen); vb = np.maximum.reduce([d.shot_value.to_numpy(), s * d.best_pass_value.to_numpy(), vc]); out.append(np.where(vb > 0, vc / vb, 1.0))
    return np.array(out)
E = effs(R); R["eff_min"], R["eff_max"] = E.min(0), E.max(0)
R["verdict_rel"] = np.where(R.eff_max < 0.7, "clearly suboptimal", np.where(R.eff_min >= 0.9, "near-optimal", "ambiguous"))
T = R[R.test]; P = T[T.kind == "pass"]
for k, d in T.groupby("kind"): print(k, (d.verdict_rel.value_counts(normalize=True) * 100).round(1).to_dict())
um = P.match.unique(); idx = {u: np.flatnonzero(P.match.to_numpy() == u) for u in um}; bs = []
for _ in range(500):
    x = P.iloc[np.concatenate([idx[u] for u in rng.choice(um, len(um))])]; bs.append(x[x.verdict_rel == "near-optimal"].realised.mean() - x[x.verdict_rel == "clearly suboptimal"].realised.mean())
g, b = P[P.verdict_rel == "near-optimal"], P[P.verdict_rel == "clearly suboptimal"]
print(f"relative verdict known groups (passes): near-optimal {g.realised.mean():.4f} (n={len(g)}) vs clearly suboptimal {b.realised.mean():.4f} (n={len(b)}); difference {g.realised.mean()-b.realised.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]")
cols = ["comp", "match", "period", "clock", "team", "event_index", "kind", "outcome", "dist_m", "v_chosen", "shot_value", "best_pass_value", "best_alt", "regret", "regret_low", "regret_high", "efficiency", "eff_min", "eff_max", "verdict", "verdict_rel"]
R[R.test][cols].round(4).to_csv(D + "decision_values_test.csv", index=False)
