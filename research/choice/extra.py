import sys
S = "research/shots/choice"
sys.argv = ["x", f"{S}/cands.pkl", f"{S}/passes_cov.pkl", f"{S}/results.pkl"]
import numpy as np, pandas as pd
import model as M
Cd, Pd = M.Cd, M.Pd.reset_index(); R = pd.read_pickle(f"{S}/results.pkl"); rule = "R0_primary"; rng = np.random.default_rng(1)
cI = lambda x: f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"
print("== END-TO-END (R0; receiver excluded or set < 2 counts as a miss) ==")
for te in (1, 2, 3):
    r = R[(rule, te)]; tot = int((Pd.game == te).sum()); n = r["n"]
    print(f"G{te}: all completed passes {tot}; modelled {n} ({n / tot:.1%}); end-to-end top-1 model {r['model_hit'].sum() / tot:.3f}, nearest-among-admissible {r['near_hit'].sum() / tot:.3f}, uniform {r['unif_hit'].sum() / tot:.3f}; MRR model {r['model_rr'].sum() / tot:.3f}, nearest {r['near_rr'].sum() / tot:.3f}, uniform {r['unif_rr'].sum() / tot:.3f}")
print("\n== BY CANDIDATE-SET SIZE (R0, pooled out-of-fold, each pass once) ==")
cat = lambda k: np.concatenate([R[(rule, te)][k] for te in (1, 2, 3)])
sz = cat("sizes"); bins = pd.cut(sz, [0, 6, 7, 8, 10], labels=["<=6", "7", "8", "9-10"])
T = pd.DataFrame({"bin": bins, "m": cat("model_hit"), "u": cat("unif_hit"), "n": cat("near_hit"), "mr": cat("model_rr"), "ur": cat("unif_rr")})
print(T.groupby("bin", observed=True).agg(n_passes=("m", "size"), model_top1=("m", "mean"), nearest_top1=("n", "mean"), uniform_top1=("u", "mean"), model_MRR=("mr", "mean"), uniform_MRR=("ur", "mean")).round(3).to_string())
print("\n== CALIBRATION of model probabilities (R0, pooled OOF; top-choice probability vs top-1 accuracy, 5 equal-count bins) ==")
mp, mh = cat("maxp"), cat("model_hit"); o = np.argsort(mp)
for ch in np.array_split(o, 5): print(f"  mean top prob {mp[ch].mean():.3f} -> accuracy {mh[ch].mean():.3f} (n={len(ch)})")
print("  log loss pooled model", round(float(cat('model_ll').mean()), 3), "uniform", round(float(cat('unif_ll').mean()), 3))
print("\n== DESCRIPTIVE CHOICE PATTERNS (R0; covered passes with >=2 candidates; actual vs what uniform choice over candidates would give) ==")
A = Cd[Cd[rule]].copy(); A = A[A["pass"].isin(A[A.is_actual]["pass"])]; A = A[A.groupby("pass")["pass"].transform("size") >= 2]
ac = A[A.is_actual].set_index("pass"); P0 = Pd.set_index("pass").loc[ac.index]
cand_fwd = A.groupby("pass").forward.apply(lambda s: (s > 0).mean()); cand_d = A.groupby("pass").dist.mean(); cand_dx = A.groupby("pass").forward.mean()
nearest = A.loc[A.groupby("pass").dist.idxmin()].set_index("pass").is_actual; safest = A.loc[A.groupby("pass").margin.idxmin()].set_index("pass").is_actual
rows = []
for g in (1, 2, 3):
    ix = ac.index[ac.game == g]
    rows.append(dict(game=g, n=len(ix), forward_share_actual=(ac.loc[ix, "forward"] > 0).mean(), forward_share_if_uniform=cand_fwd[ix].mean(), median_dist_actual=ac.loc[ix, "dist"].median(), mean_dist_actual=ac.loc[ix, "dist"].mean(),
                     mean_dist_candidates=cand_d[ix].mean(), mean_forward_actual=ac.loc[ix, "forward"].mean(), mean_forward_candidates=cand_dx[ix].mean(), median_rec_pressure=ac.loc[ix, "rec_press"].median(), share_nearest=nearest[ix].mean(), share_safest_lane=safest[ix].mean(), median_margin_actual=ac.loc[ix, "margin"].median()))
print(pd.DataFrame(rows).round(3).T.to_string(header=False))
P0["tert"] = pd.qcut(P0.p_press, 3, labels=["high pressure (nearest opp. closest third)", "medium", "low pressure"]); P0["fwd"] = ac.loc[P0.index, "forward"] > 0; P0["dist"] = ac.loc[P0.index, "dist"]
print(f"\n  passer pressure terciles (nearest-opponent distance, pooled cut-offs {np.quantile(P0.p_press, [1/3, 2/3]).round(2)} m): forward share / median receiver distance")
for t, g in P0.groupby("tert", observed=True): print(f"   {t}: n={len(g)}, forward {g.fwd.mean():.1%}, median distance {g.dist.median():.1f} m; by game forward: " + ", ".join(f"G{k}: {gg.fwd.mean():.1%}" for k, gg in g.groupby("game")))
P0["zone"] = pd.cut(P0.px, [-60, -35 / 2, 35 / 2, 60], labels=["defensive third", "middle third", "attacking third"])
print("  by passer pitch third: " + "; ".join(f"{z}: n={len(g)}, forward {g.fwd.mean():.1%}, median dist {g.dist.median():.1f} m" for z, g in P0.groupby("zone", observed=True)))
print("  by passer role proxy: " + "; ".join(f"{z}: n={len(g)}, forward {g.fwd.mean():.1%}, median dist {g.dist.median():.1f} m" for z, g in P0.groupby("role")))
print("\n== PER-GAME CONDITIONAL-LOGIT COEFFICIENTS (descriptive association; standardised with pooled R0 candidates; ridge 1; 200 bootstrap resamples over passes) ==")
prep = M.Prep().fit(A[M.FEATS].to_numpy(float))
def coefs(games, boots=200):
    d, st = M.sets_for(rule, games); X = prep.transform(d[M.FEATS].to_numpy(float)); b0 = M.fit_clogit(X, M.chosen_idx(d, st), st, 1.0)
    sizes = np.diff(np.r_[st, len(d)]); out = []
    for _ in range(boots):
        pick = rng.integers(0, len(st), len(st)); rows_ = np.concatenate([np.arange(st[i], st[i] + sizes[i]) for i in pick]); stb = np.r_[0, np.cumsum(sizes[pick])[:-1]]
        Xb = X[rows_]; chb = np.flatnonzero(d.is_actual.to_numpy()[rows_]); out.append(M.fit_clogit(Xb, chb, stb, 1.0))
    out = np.array(out); return b0, np.percentile(out, 2.5, 0), np.percentile(out, 97.5, 0)
for nm, gs in (("G1", [1]), ("G2", [2]), ("G3", [3]), ("all", [1, 2, 3])):
    b0, lo, hi = coefs(gs); print(f"  {nm}: " + "; ".join(f"{f} {b:+.2f} [{l:+.2f},{h:+.2f}]" for f, b, l, h in zip(M.FEATS, b0, lo, hi)))
print("\n== POST-HOC EXTENSION (not pre-registered; chosen after seeing that the velocity-aligned rule rivals the model): primary + moving*cos(passer velocity, direction to candidate) ==")
Cd["velcos"] = np.where(Cd["pass"].map(Pd.set_index("pass").speed) >= 1.0, Cd.cos_vel, 0.0)
M.FEATS = M.FEATS + ["velcos"]; M.Cd = Cd
for te in (1, 2, 3):
    tr = [g for g in (1, 2, 3) if g != te]; r2 = M.evaluate(rule, tr, te); r1 = R[(rule, te)]; n = r2["n"]; idx = rng.integers(0, n, (1000, n))
    dm = lambda a, b: np.array([(a - b)[i].mean() for i in idx])
    print(f"  held-out G{te}: M1 top-1 {r1['model_hit'].mean():.3f} / MRR {r1['model_rr'].mean():.3f} / LL {r1['model_ll'].mean():.3f}  ->  M2 top-1 {r2['model_hit'].mean():.3f} / MRR {r2['model_rr'].mean():.3f} / LL {r2['model_ll'].mean():.3f}; paired top-1 {dm(r2['model_hit'], r1['model_hit']).mean():+.3f} {cI(dm(r2['model_hit'], r1['model_hit']))}, LL {dm(r2['model_ll'], r1['model_ll']).mean():+.3f} {cI(dm(r2['model_ll'], r1['model_ll']))}; M2 coefficient on velcos (standardised) {r2['beta'][-1]:+.2f}")
