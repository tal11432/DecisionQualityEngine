"""DQ-v14/v15 validation of the Decision Score (PREREG_DQ_v13.txt amendments A-E): the ABSOLUTE scale of amendment E. Scores every test decision (shots, ground passes and carries at x >= 80) and runs the gates (G-resolution is withdrawn; face-validity tests are in tests/)."""
import os, sys, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); rng = np.random.default_rng(0)
from dscore import score as S
from dscore.core import Situation
M13 = S.v13(); Dd = pd.read_pickle(f"{D}/v14_dec.pkl")
def sit(r, drop=None):
    mates = list(r.mates)
    if drop is not None: mates = [m for i, m in enumerate(mates) if i not in drop]
    return Situation([r.ax, r.ay], np.array(r.opps, float).reshape(-1, 2), mates, r.gk)
def chosen(r): return ("shot",) if r.kind == "shot" else (r.kind, [r.tx, r.ty])
def score_row(r, scales=None):
    j = S.judge_both(sit(r), chosen(r), scales=scales); jd = S.judge(sit(r), chosen(r), include_doubt=True, scales=scales)
    return dict(score_chosen=j["score_chosen"], score_best=j["score_best"], gap_points=j["gap_points"], gap_xg=j["gap_xg"], q_chosen=j["q_chosen"], q_best=j["q_best"], score_worst=j["score_worst"], best_kind=j["best"]["kind"], gap_points_doubt=jd["gap_points"], q_best_doubt=jd["q_best"],
                q_chosen_model=j["q_chosen_model"], q_best_model=j["q_best_model"], score_chosen_model=j["score_chosen_model"], score_best_model=j["score_best_model"], gap_xg_model=j["gap_xg_model"], gap_points_model=j["gap_points_model"],
                counter_assumed=j["counter_assumed_chosen"], counter_assumed_best=j["counter_assumed_best"], counter_assumed_any=j["counter_assumed_any"])
def chosen_q(r):
    s_ = sit(r)
    if r.kind == "shot": return S.core.shot_value(s_)[2]
    return (S.pass_value(s_, [r.tx, r.ty]) if r.kind == "pass" else S.carry_value(s_, [r.tx, r.ty]))["v"]
REF = f"{D}/v14_reference.pkl"
if not os.path.exists(REF):
    tr = Dd[~Dd.test]; smp_ref = tr.sample(9000, random_state=0)      # a random sample (natural mix of shots, passes and carries), as pre-registered
    pd.to_pickle(dict(q=np.array([chosen_q(r) for r in smp_ref.itertuples()]), kinds=list(smp_ref.kind), n=len(smp_ref)), REF); print("reference built:", len(smp_ref), "decisions")
ref = pd.read_pickle(REF); S.set_reference(ref["q"]); print(f"reference: {len(ref['q'])} decisions of the training competitions; Q of the options taken, 10th/50th/90th/99th pct: " + ", ".join(f"{np.percentile(ref['q'], q):+.4f}" for q in (10, 50, 90, 99)))
RAW = f"{D}/v14_scored_raw.pkl"
if os.path.exists(RAW): T = pd.read_pickle(RAW); print("loaded cached scoring:", len(T))
else:
    T = Dd[Dd.test].reset_index(drop=True); R = pd.DataFrame([score_row(r) for r in T.itertuples()]); T = pd.concat([T.drop(columns=["opps", "mates", "gk"]), R], axis=1); T.to_pickle(RAW)
print(f"test decisions {len(T)}: {dict(T.kind.value_counts())}; competitions {dict(T.comp.value_counts())}")
T["score_chosen"] = [S.q_to_score(q) for q in T.q_chosen]; T["score_best"] = [S.q_to_score(q) for q in T.q_best]; T["gap_points"] = T.score_best - T.score_chosen      # ABSOLUTE scale (amendment E), from Q only
print("\n== MARKED ASSUMPTION A1 (visible balance >= 2 at a loss point with x <= 70: loss cost >= C): how much it touches the test decisions")
dif = (T.score_chosen != T.score_chosen_model); gdif = (T.gap_points != T.gap_points_model)
print(f"   options flagged somewhere in the moment: {T.counter_assumed_any.mean()*100:.1f}% of decisions; the option TAKEN is flagged in {T.counter_assumed.sum()} decisions ({T.counter_assumed.mean()*100:.2f}%); the BEST option is flagged in {T.counter_assumed_best.sum()}")
print(f"   score of the option taken changes in {int(dif.sum())} decisions ({dif.mean()*100:.2f}%), by a mean of {abs(T.score_chosen-T.score_chosen_model)[dif].mean() if dif.any() else 0:.1f} points when it changes; the gap changes in {int(gdif.sum())} decisions ({gdif.mean()*100:.2f}%); by kind of the option taken, flagged: {dict(T[T.counter_assumed].kind.value_counts())}")
print(f"   gap >= noise floor (0.03 xG): with A1 {(T.gap_xg>=0.03).mean()*100:.1f}% vs model-only {(T.gap_xg_model>=0.03).mean()*100:.1f}%")
T["pct_chosen"] = [S.rank_score(q) for q in T.q_chosen]; T["pct_best"] = [S.rank_score(q) for q in T.q_best]                                          # 'rarity' column (percentile among the reference decisions): NOT the Decision Score
T["realised"] = T.own10 - T.opp10 + np.where(T.kind == "shot", T.sb_xg.fillna(0), 0.0)
print(f"\n== DISTRIBUTION of the score of the option taken (ABSOLUTE scale, C = {S.v15()['C']:.3f}), by kind (test); no resolution gate (withdrawn in amendment E)")
print(f"   pooled 10th/50th/90th/99th {np.percentile(T.score_chosen,10):.0f}/{np.percentile(T.score_chosen,50):.0f}/{np.percentile(T.score_chosen,90):.0f}/{np.percentile(T.score_chosen,99):.0f}; share <= 10: {(T.score_chosen<=10).mean()*100:.1f}%, >= 50: {(T.score_chosen>=50).mean()*100:.1f}%, >= 80: {(T.score_chosen>=80).mean()*100:.1f}%")
for k, g in T.groupby("kind"): print(f"   {k}: n={len(g)}, score_chosen mean {g.score_chosen.mean():.1f} (10th/50th/90th {np.percentile(g.score_chosen,10):.0f}/{np.percentile(g.score_chosen,50):.0f}/{np.percentile(g.score_chosen,90):.0f}); gap points mean {g.gap_points.mean():.1f}, median {g.gap_points.median():.0f}, share with gap 0 {(g.gap_points==0).mean()*100:.0f}%, gap > 10 pts {(g.gap_points>10).mean()*100:.0f}%; best option kind {dict(g.best_kind.value_counts(normalize=True).round(2))}")
# ---------------- G-calibration
def sp(x, y): return float(pd.Series(x).rank().corr(pd.Series(y).rank()))
print("\n== G-calibration: predicted Q of the option taken vs realised net xG of the next 10 s (own - opponent; + the shot's own xG for shots)")
for lab, g in [("ALL", T)] + [(k, g) for k, g in T.groupby("kind")] + [("passes completed", T[(T.kind == "pass") & (T.completed == 1)]), ("passes failed (target = where the ball stopped)", T[(T.kind == "pass") & (T.completed == 0)])]:
    print(f"   {lab}: n={len(g)}, mean Q {g.q_chosen.mean():+.4f}, mean realised {g.realised.mean():+.4f}, ratio {g.q_chosen.mean()/g.realised.mean() if abs(g.realised.mean())>1e-9 else float('nan'):.2f}")
T["dec"] = pd.qcut(T.score_chosen.rank(method="first"), 10, labels=False); dm = T.groupby("dec").agg(score=("score_chosen", "mean"), q=("q_chosen", "mean"), r=("realised", "mean"), n=("realised", "size"))
print("   deciles of score_chosen: " + "; ".join(f"D{i+1}: score {row.score:.0f}, Q {row.q:+.3f}, realised {row.r:+.3f}" for i, row in dm.iterrows()))
um = T.match.unique(); idx = {u: np.flatnonzero(T.match.to_numpy() == u) for u in um}; sps = []
for _ in range(300):
    x = T.iloc[np.concatenate([idx[u] for u in rng.choice(um, len(um))])]; d2 = x.assign(dec=pd.qcut(x.score_chosen.rank(method="first"), 10, labels=False)).groupby("dec").agg(s=("score_chosen", "mean"), r=("realised", "mean")); sps.append(sp(d2.s, d2.r))
rho = sp(dm.score, dm.r); ratio = T.q_chosen.mean() / T.realised.mean()
nf = T[~((T.kind == "pass") & (T.completed == 0))]; ratio_nf = nf.q_chosen.mean() / nf.realised.mean(); print(f"   without failed passes: n={len(nf)}, mean Q {nf.q_chosen.mean():+.4f}, mean realised {nf.realised.mean():+.4f}, ratio {ratio_nf:.2f}")
print(f"   Spearman across deciles {rho:.2f} [{np.percentile(sps,2.5):.2f},{np.percentile(sps,97.5):.2f}]; pooled mean Q / mean realised = {ratio:.2f}  => G-calibration (all) {'PASS' if rho >= 0.9 and 0.85 <= ratio <= 1.15 else 'FAIL'}; without failed passes {'PASS' if rho >= 0.9 and 0.85 <= ratio_nf <= 1.15 else 'FAIL'}")
# ---------------- G-gap
print("\n== G-gap: inside terciles of Q_best, realised value of the top vs bottom tercile of the gap (expected: top gap realises LESS)")
T["qb"] = pd.qcut(T.q_best.rank(method="first"), 3, labels=["low promise", "mid", "high promise"]); okc = 0; right = 0
for lab, g in T.groupby("qb", observed=True):
    g = g.assign(gtile=pd.qcut(g.gap_xg.rank(method="first"), 3, labels=False)); hi, lo = g[g.gtile == 2], g[g.gtile == 0]; diff = hi.realised.mean() - lo.realised.mean(); bs = []
    um2 = g.match.unique(); id2 = {u: np.flatnonzero(g.match.to_numpy() == u) for u in um2}
    for _ in range(300):
        x = g.iloc[np.concatenate([id2[u] for u in rng.choice(um2, len(um2))])]; bs.append(x[x.gtile == 2].realised.mean() - x[x.gtile == 0].realised.mean())
    lo_, hi_ = np.percentile(bs, [2.5, 97.5]); right += diff < 0; okc += hi_ < 0
    print(f"   {lab}: n={len(g)}; realised top gap {hi.realised.mean():+.4f} (mean gap {hi.gap_xg.mean():.3f} xG) vs bottom gap {lo.realised.mean():+.4f} (mean gap {lo.gap_xg.mean():.3f}); difference {diff:+.4f} [{lo_:+.4f},{hi_:+.4f}]")
print(f"   direction right in {right}/3 strata, CI excluding 0 in {okc}/3  => G-gap {'PASS' if right == 3 and okc >= 2 else 'FAIL'}")
# ---------------- G-best-sanity
cl = T[(T.kind == "shot") & (T.q_chosen >= 0.30)]; print(f"\n== G-best-sanity: shots with Q >= 0.30 (clear chances) n={len(cl)}; median gap {cl.gap_points.median():.0f} points, mean {cl.gap_points.mean():.1f}; share with gap 0 {(cl.gap_points==0).mean()*100:.0f}%  => {'PASS' if cl.gap_points.median() <= 10 else 'FAIL'}")
# ---------------- G-curse
g0 = T[T.gap_points == 0]; print(f"== G-curse: decisions where the option taken is the modelled best (n={len(g0)}): mean Q {g0.q_chosen.mean():+.4f} vs realised {g0.realised.mean():+.4f} (ratio {g0.q_chosen.mean()/g0.realised.mean():.2f}) => {'ok' if 0.8 <= g0.q_chosen.mean()/g0.realised.mean() <= 1.25 else 'FLAG: optimistic best option'}")
# ---------------- G-anchor (model level)
print("\n== G-anchor: the scale and modelled counter-attack situations (the face-validity unit tests are in tests/test_dscore_score.py)")
print(f"   share of the options taken that score <= 5: {(T.score_chosen<=5).mean()*100:.2f}%; score 100: {(T.score_chosen==100).mean()*100:.2f}%")
print("   A1 is ON in every number above unless a column says _model. scale: Q=1.0 ->", S.q_to_score(1.0), "| 0.70 ->", S.q_to_score(0.70), "| 0.32 ->", S.q_to_score(0.32), "| 0.08 ->", S.q_to_score(0.08), "| 0.06 ->", S.q_to_score(0.06), "| 0.01 ->", S.q_to_score(0.01), "| 0 ->", S.q_to_score(0.0), "| -0.012 ->", S.q_to_score(-0.012), "| -C ->", S.q_to_score(-S.v15()['C']))
def lg_(a, O, T_, k="pass"): return S.loss_danger(np.array(a, float), np.array(O, float).reshape(-1, 2), np.array(T_, float).reshape(-1, 2), k)
cases = [("a loss at the edge of our box, 3 opponents beyond the ball, 1 of ours covering", [18.0, 40.0], [[10.0, 36.0], [12.0, 44.0], [14.0, 40.0], [40.0, 40.0]], [[4.0, 40.0], [60.0, 30.0]]),
         ("the same loss, nobody of theirs beyond the ball, 3 of ours covering", [18.0, 40.0], [[40.0, 40.0], [45.0, 30.0]], [[4.0, 40.0], [10.0, 30.0], [12.0, 50.0]]),
         ("a loss in the opponent box, 3 of ours covering", [105.0, 40.0], [[112.0, 40.0], [100.0, 20.0]], [[80.0, 40.0], [70.0, 30.0], [60.0, 50.0]])]
for lab, a, O_, T_ in cases: l = lg_(a, O_, T_); print(f"   {lab}: L = {l:+.4f} xG -> score of the turnover {S.q_to_score(-l)}")
# ---------------- stability and visibility on a sample
CUTS = np.percentile(T.score_chosen, [33.3, 66.7]); smp = T.sample(min(250, len(T)), random_state=0).index; src = Dd[Dd.test].reset_index(drop=True); widths, kept, vis, gw = [], [], [], []
for i in smp:
    r = next(src.iloc[[i]].itertuples()); base = score_row(r); sc = []; gp = []
    for _ in range(60):
        r2 = r._replace(ax=r.ax + rng.normal(0, 0.8) * 120 / 105, ay=r.ay + rng.normal(0, 0.8) * 80 / 68, opps=[np.array(o) + rng.normal(0, 0.8, 2) * [120 / 105, 80 / 68] for o in r.opps], mates=[np.array(m) + rng.normal(0, 0.8, 2) * [120 / 105, 80 / 68] for m in r.mates])
        if r.kind != "shot": r2 = r2._replace(tx=r.tx + rng.normal(0, 0.8) * 120 / 105, ty=r.ty + rng.normal(0, 0.8) * 80 / 68)
        o = score_row(r2, scales=dict(shot=float(np.exp(rng.normal(0, 0.10))), pass_=1.0) if False else {"shot": float(np.exp(rng.normal(0, 0.10))), "pass": float(np.exp(rng.normal(0, 0.09))), "carry": float(np.exp(rng.normal(0, 0.178)))}); sc.append(o["score_chosen"]); gp.append(o["gap_points"])
    widths.append(np.percentile(sc, 90) - np.percentile(sc, 10)); gw.append(np.percentile(gp, 90) - np.percentile(gp, 10)); kept.append(np.mean(np.digitize(sc, CUTS) == np.digitize(base["score_chosen"], CUTS)))
    if len(r.mates) >= 2:
        d = set(rng.choice(len(r.mates), len(r.mates) // 2, replace=False).tolist()); sd = sit(r, drop=d); ch = chosen(r); jj = S.judge(sd, ch); vis.append(abs(jj["gap_points"] - base["gap_points"]))
print(f"\n== G-stability (n={len(widths)} decisions, 60 perturbations each; 0.8 m position noise, value scales 9%/10%/18%): median band width (10th-90th pct) of the chosen score {np.median(widths):.1f} points, of the gap {np.median(gw):.1f}; share of decisions that keep their score tercile in >= 80% of draws {np.mean(np.array(kept)>=0.8)*100:.0f}%  => G-stability {'PASS' if np.median(widths) <= 25 and np.mean(np.array(kept)>=0.8) >= 0.70 else 'FAIL'}")
print(f"== G-visibility: hiding half of the visible teammates changes the gap by a median of {np.median(vis):.1f} points (n={len(vis)})")
T.drop(columns=["dec"]).to_pickle(f"{D}/v14_scored.pkl")
out = T[["comp", "match", "period", "minute", "second", "team", "index", "kind", "ax", "ay", "score_chosen", "score_best", "gap_points", "gap_xg", "q_chosen", "q_best", "pct_chosen", "pct_best", "best_kind", "score_chosen_model", "gap_points_model", "counter_assumed"]].copy()
out["clock"] = out.minute.astype(int).astype(str).str.zfill(2) + ":" + out.second.astype(int).astype(str).str.zfill(2); out.drop(columns=["minute", "second"]).round(4).to_csv(f"{D}/decision_scores_test.csv", index=False); print("\nwrote decision_scores_test.csv with", len(out), "rows (no player names)")
