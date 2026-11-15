"""DQ-v16 validation of the whole-pitch Decision Score (PREREG_DQ_v13.txt amendment J). Scores every test decision (foot shots at x >= 60, ground passes and carries anywhere) with the absolute scale of amendment E, the marked assumption A1 (primary) and the model-only values;
gates per zone on the MODEL-ONLY value, A1 effect, range report, named probes, no-regression check in the attacking third."""
import os, sys, subprocess, io, numpy as np, pandas as pd
from multiprocessing import Pool
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); rng = np.random.default_rng(0)
from dscore import score as S, core
from dscore.core import Situation
Dd = pd.read_pickle(f"{D}/v16_dec.pkl"); Dd = Dd[Dd.test].reset_index(drop=True); ZN = ["own third", "middle", "attacking"]; Dd["zone"] = [core.zone(x) for x in Dd.ax]
def sit(r): return Situation([r.ax, r.ay], r.opps.astype(float), list(r.mates.astype(float)), r.gk)
def chosen(r): return ("shot",) if r.kind == "shot" else (r.kind, [r.tx, r.ty])
def score_row(r, scales=None):
    j = S.judge_both(sit(r), chosen(r), scales=scales); jd = S.judge(sit(r), chosen(r), include_doubt=True, scales=scales) if scales is None else None
    real = [o["score"] for o in j["options"] if o["kind"] != "turnover"]
    return dict(score_chosen=j["score_chosen"], score_best=j["score_best"], gap_points=j["gap_points"], gap_xg=j["gap_xg"], q_chosen=j["q_chosen"], q_best=j["q_best"], score_worst=j["score_worst"], score_worst_real=min(real) if real else np.nan, best_kind=j["best"]["kind"],
                gap_points_doubt=jd["gap_points"] if jd else np.nan, q_chosen_model=j["q_chosen_model"], q_best_model=j["q_best_model"], score_chosen_model=j["score_chosen_model"], score_best_model=j["score_best_model"], gap_xg_model=j["gap_xg_model"], gap_points_model=j["gap_points_model"],
                counter_assumed=j["counter_assumed_chosen"], counter_assumed_best=j["counter_assumed_best"], counter_assumed_any=j["counter_assumed_any"], n_options=len(j["options"]))
def work(i): return score_row(next(Dd.iloc[[i]].itertuples()))
def perturb(args):
    i, seed = args; r = next(Dd.iloc[[i]].itertuples()); g = np.random.default_rng(seed); k = np.array([120 / 105, 80 / 68])
    r2 = r._replace(ax=r.ax + g.normal(0, 0.8) * k[0], ay=r.ay + g.normal(0, 0.8) * k[1], opps=np.array(r.opps, float) + g.normal(0, 0.8, (len(r.opps), 2)) * k, mates=np.array(r.mates, float) + g.normal(0, 0.8, (len(r.mates), 2)) * k)
    if r.kind != "shot": r2 = r2._replace(tx=r.tx + g.normal(0, 0.8) * k[0], ty=r.ty + g.normal(0, 0.8) * k[1])
    o = score_row(r2, scales={"shot": float(np.exp(g.normal(0, 0.10))), "pass": float(np.exp(g.normal(0, 0.09))), "carry": float(np.exp(g.normal(0, 0.178)))}); return o["score_chosen"], o["gap_points"]
if __name__ == "__main__":
    t_start = pd.Timestamp.now(); print(f"test decisions {len(Dd)}: {dict(Dd.kind.value_counts())}; by zone {dict(Dd.zone.value_counts())}; competitions {dict(Dd.comp.value_counts())}", flush=True)
    RAW = f"{D}/v16_scored_raw.pkl"
    if os.path.exists(RAW): R = pd.read_pickle(RAW).to_dict("records"); print("loaded cached scoring", len(R))
    else:
        with Pool(4) as pool: R = pool.map(work, range(len(Dd)), chunksize=200)
        pd.DataFrame(R).to_pickle(RAW)
    T = pd.concat([Dd.drop(columns=["opps", "mates", "gk"]), pd.DataFrame(R)], axis=1); mins = (pd.Timestamp.now() - t_start).total_seconds() / 60; print(f"scored {len(T)} decisions in {mins:.1f} min", flush=True)
    T["realised"] = T.own10 - T.opp10 + np.where(T.kind == "shot", T.sb_xg.fillna(0), 0.0); T["pct_chosen"] = np.nan
    def sp(x, y): return float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    def pr(lab, g): return f"{lab}: n={len(g)}, mean Q {g.q_chosen_model.mean():+.4f}, realised {g.realised.mean():+.4f}"
    # ---------------- distribution and RANGE REPORT
    print("\n== RANGE REPORT (descriptive, requested by the user): score of the options taken, by zone (A1 on = primary)")
    qs = [0.1, 1, 5, 50, 95, 99, 99.9]
    for z, g in [("ALL", T)] + [(z, T[T.zone == z]) for z in ZN]:
        print(f"   {z}: n={len(g)}; percentiles {', '.join(f'{q}%: {np.percentile(g.score_chosen, q):.0f}' for q in qs)}; min {g.score_chosen.min()}, max {g.score_chosen.max()}; model-only 0.1%/50%/99.9%: {np.percentile(g.score_chosen_model, 0.1):.0f}/{np.percentile(g.score_chosen_model, 50):.0f}/{np.percentile(g.score_chosen_model, 99.9):.0f}")
    print("   by kind: " + "; ".join(f"{k}: 1%/50%/99% {np.percentile(g.score_chosen, 1):.0f}/{np.percentile(g.score_chosen, 50):.0f}/{np.percentile(g.score_chosen, 99):.0f}" for k, g in T.groupby("kind")))
    print("== the WORST real option (a pass or carry to somewhere bad) available in each moment, excluding the hypothetical unforced turnover: " + "; ".join(f"{z}: 1%/5%/50% {np.nanpercentile(T[T.zone==z].score_worst_real, 1):.0f}/{np.nanpercentile(T[T.zone==z].score_worst_real, 5):.0f}/{np.nanpercentile(T[T.zone==z].score_worst_real, 50):.0f}" for z in ZN) + f"; incl. the unforced turnover, 1%/50%: {np.percentile(T.score_worst, 1):.0f}/{np.percentile(T.score_worst, 50):.0f}")
    # ---------------- A1 effect
    print("\n== MARKED ASSUMPTION A1 by zone (visible balance >= 2 at a loss point with x <= 70 costs at least C): effect on the test decisions")
    for z, g in [("ALL", T)] + [(z, T[T.zone == z]) for z in ZN]:
        dif = g.score_chosen != g.score_chosen_model; gd = g.gap_points != g.gap_points_model
        print(f"   {z}: some option flagged in {g.counter_assumed_any.mean()*100:.1f}% of decisions; the option TAKEN flagged in {int(g.counter_assumed.sum())} ({g.counter_assumed.mean()*100:.2f}%); best flagged {int(g.counter_assumed_best.sum())}; score of the option taken changes in {dif.mean()*100:.2f}% (mean {abs(g.score_chosen-g.score_chosen_model)[dif].mean() if dif.any() else 0:.1f} points when it changes); gap changes in {gd.mean()*100:.2f}%; gap >= 0.03 xG: A1 {(g.gap_xg>=0.03).mean()*100:.1f}% vs model-only {(g.gap_xg_model>=0.03).mean()*100:.1f}%")
    fl = T[T.counter_assumed]; print(f"   flagged options TAKEN: predicted Q with A1 {fl.q_chosen.mean() if len(fl) else float('nan'):+.4f} vs model-only {fl.q_chosen_model.mean() if len(fl) else float('nan'):+.4f} vs realised {fl.realised.mean() if len(fl) else float('nan'):+.4f} (n={len(fl)}; the assumption is expected to disagree with the realised value)")
    # ---------------- G-calibration per zone, MODEL-ONLY Q
    print("\n== G-calibration per zone (MODEL-ONLY Q of the option taken vs realised net xG of the next 10 s; in a zone with |mean realised| < 0.01 the ratio is replaced by |mean Q - mean realised| <= 0.005)")
    cal_ok = {}
    for z, g in [("ALL", T)] + [(z, T[T.zone == z]) for z in ZN]:
        for lab, h in (("all decisions", g), ("without failed passes", g[~((g.kind == "pass") & (g.completed == 0))])):
            mq, mr = h.q_chosen_model.mean(), h.realised.mean(); dm = h.assign(dec=pd.qcut(h.q_chosen_model.rank(method="first"), 10, labels=False)).groupby("dec").agg(q=("q_chosen_model", "mean"), r=("realised", "mean")); rho = sp(dm.q, dm.r)
            ratio_ok = (abs(mq - mr) <= 0.005) if abs(mr) < 0.01 else (0.85 <= mq / mr <= 1.15); ok = ratio_ok and rho >= 0.9; cal_ok[(z, lab)] = ok
            print(f"   {z} / {lab}: n={len(h)}, mean Q {mq:+.4f}, realised {mr:+.4f}, ratio {mq/mr if abs(mr)>1e-9 else float('nan'):.2f}, Spearman over deciles {rho:.2f} => {'PASS' if ok else 'FAIL'}")
    for z in ZN:
        for k, g in T[T.zone == z].groupby("kind"): print(f"      {z}, {pr(k, g)}, ratio {g.q_chosen_model.mean()/g.realised.mean() if abs(g.realised.mean())>1e-9 else float('nan'):.2f}")
    # ---------------- G-gap per zone, MODEL-ONLY
    print("\n== G-gap per zone: inside terciles of Q_best (model-only), realised value of the top vs bottom tercile of the gap (expected: top gap realises LESS)")
    for z, g0 in [("ALL", T)] + [(z, T[T.zone == z]) for z in ZN]:
        g0 = g0.assign(qb=pd.qcut(g0.q_best_model.rank(method="first"), 3, labels=False)); right = okc = 0; parts = []
        for q, g in g0.groupby("qb"):
            g = g.assign(gtile=pd.qcut(g.gap_xg_model.rank(method="first"), 3, labels=False)); hi, lo = g[g.gtile == 2], g[g.gtile == 0]; diff = hi.realised.mean() - lo.realised.mean(); um = g.match.unique(); idx = {u: np.flatnonzero(g.match.to_numpy() == u) for u in um}; bs = []
            for _ in range(300):
                x = g.iloc[np.concatenate([idx[u] for u in rng.choice(um, len(um))])]; bs.append(x[x.gtile == 2].realised.mean() - x[x.gtile == 0].realised.mean())
            l_, h_ = np.percentile(bs, [2.5, 97.5]); right += diff < 0; okc += h_ < 0; parts.append(f"tercile {q+1}: top gap {hi.realised.mean():+.4f} (mean gap {hi.gap_xg_model.mean():.3f}) vs bottom {lo.realised.mean():+.4f}; diff {diff:+.4f} [{l_:+.4f},{h_:+.4f}]")
        print(f"   {z}: direction right {right}/3, CI excluding 0 in {okc}/3 => {'PASS' if right == 3 and okc >= 2 else 'FAIL'}   |   " + " | ".join(parts))
    # ---------------- G-best-sanity, G-curse
    cl = T[(T.kind == "shot") & (T.q_chosen_model >= 0.30)]; print(f"\n== G-best-sanity: shots with Q >= 0.30 n={len(cl)}; median gap {cl.gap_points_model.median():.0f} points; share with gap 0 {(cl.gap_points_model==0).mean()*100:.0f}% => {'PASS' if cl.gap_points_model.median() <= 10 else 'FAIL'}")
    g0 = T[T.gap_points_model == 0]; print(f"== G-curse: decisions where the option taken is the modelled best (n={len(g0)}): mean Q {g0.q_chosen_model.mean():+.4f} vs realised {g0.realised.mean():+.4f} (ratio {g0.q_chosen_model.mean()/g0.realised.mean():.2f})")
    print("== noise floor: gap >= 0.03 xG by zone (model-only): " + "; ".join(f"{z}: {(T[T.zone==z].gap_xg_model>=0.03).mean()*100:.1f}%" for z in ZN) + f"; by kind: " + "; ".join(f"{k}: {(g.gap_xg_model>=0.03).mean()*100:.1f}%" for k, g in T.groupby("kind")))
    # ---------------- G-no-regression (attacking third) against the committed values of commit 5de0656
    try:
        old = pd.read_csv(io.StringIO(subprocess.check_output(["git", "show", "5de0656:research/dq/decision_scores_test.csv"], cwd=D, text=True)))
        m = T.merge(old[["match", "index", "kind", "q_chosen"]].rename(columns={"q_chosen": "q_old"}), on=["match", "index", "kind"], how="inner"); m = m[m.ax >= 80]
        print(f"\n== G-no-regression (x >= 80): {len(m)} decisions matched with commit 5de0656; Spearman new vs old Q of the option taken {sp(m.q_chosen_model, m.q_old):.3f}; mean |difference| {np.abs(m.q_chosen_model - m.q_old).mean():.4f} xG; mean Q old {m.q_old.mean():+.4f} new {m.q_chosen_model.mean():+.4f}; by kind: " + "; ".join(f"{k}: Spearman {sp(g.q_chosen_model, g.q_old):.3f}, mean |diff| {np.abs(g.q_chosen_model-g.q_old).mean():.4f}" for k, g in m.groupby("kind")))
        print(f"   => {'PASS' if sp(m.q_chosen_model, m.q_old) >= 0.95 and np.abs(m.q_chosen_model - m.q_old).mean() <= 0.01 and cal_ok[('attacking', 'without failed passes')] else 'FAIL'} (needs Spearman >= 0.95, mean |difference| <= 0.01 and the calibration gate in the attacking zone)")
    except Exception as ex: print("\n== G-no-regression: could not read the committed file:", ex)
    # ---------------- G-stability per zone (sample of 250 per zone, 60 perturbations)
    print("\n== G-stability per zone (250 decisions, 60 perturbations each: 0.8 m positions; value scales 9% pass, 10% shot, 18% carry)")
    for z in ZN:
        g = T[T.zone == z]; smp = g.sample(min(250, len(g)), random_state=0).index; cuts = np.percentile(g.score_chosen, [33.3, 66.7]); w = []; kept = []; gw = []
        with Pool(4) as pool: res = pool.map(perturb, [(i, 1000 * int(i) + k) for i in smp for k in range(60)], chunksize=60)
        arr = np.array(res).reshape(len(smp), 60, 2)
        for n_, i in enumerate(smp): sc = arr[n_, :, 0]; w.append(np.percentile(sc, 90) - np.percentile(sc, 10)); gw.append(np.percentile(arr[n_, :, 1], 90) - np.percentile(arr[n_, :, 1], 10)); kept.append(np.mean(np.digitize(sc, cuts) == np.digitize(T.score_chosen[i], cuts)))
        print(f"   {z}: median band width of the score {np.median(w):.1f} points, of the gap {np.median(gw):.1f}; {np.mean(np.array(kept)>=0.8)*100:.0f}% keep their score tercile in >= 80% of draws => {'PASS' if np.median(w) <= 25 and np.mean(np.array(kept)>=0.8) >= 0.70 else 'FAIL'}", flush=True)
    # ---------------- NAMED PROBES (synthetic)
    print("\n== NAMED PROBES (synthetic situations, StatsBomb units; A1 on / model-only)")
    def both(s, ch):
        j = S.judge_both(s, ch); return f"score {j['score_chosen']} / {j['score_chosen_model']} (Q {j['q_chosen']:+.3f} / {j['q_chosen_model']:+.3f}); best option {j['best']['kind']} score {j['score_best']}; worst real option {min(o['score'] for o in j['options'] if o['kind']!='turnover') if len(j['options'])>1 else None}"
    own = Situation([14.0, 40.0], np.array([[18.0, 41.0], [24.0, 36.0], [24.0, 46.0], [30.0, 40.0], [60.0, 20.0], [60.0, 60.0], [118.0, 40.0]]), [[8.0, 30.0], [8.0, 50.0], [40.0, 40.0], [45.0, 20.0]], [118.0, 40.0])
    print("   pass into the opponent's feet in our own box (target (18, 41) next to an opponent at (18, 41)):", both(own, ("pass", [18.0, 41.0])))
    print("   safe 6 m pass to a free centre-back in our own box (target (8, 30)):", both(own, ("pass", [8.0, 30.0])))
    print("   carry 5 m straight into three opponents in our own box:", both(own, ("carry", [19.0, 40.0])))
    sure = Situation([112.0, 40.0], np.array([[118.0, 52.0], [100.0, 25.0], [95.0, 55.0]]), [[108.0, 36.0], [105.0, 44.0]], [118.0, 52.0])
    print("   shot at an almost open goal from 8 units with the keeper out of position:", both(sure, ("shot",)))
    print("   safe 4 m pass to a teammate who has the goal open (teammate at (116, 40)):", both(Situation([112.0, 38.0], np.array([[118.0, 52.0], [100.0, 25.0], [95.0, 55.0]]), [[116.0, 40.0]], [118.0, 52.0]), ("pass", [116.0, 40.0])))
    T.drop(columns=["pct_chosen"]).to_pickle(f"{D}/v16_scored.pkl")
    smp = pd.concat([g.sample(min(len(g), int(30000 * len(g) / len(T)) + 1), random_state=0) for _, g in T.groupby(["zone", "kind"])])
    out = smp[["comp", "match", "period", "minute", "second", "team", "index", "kind", "zone", "ax", "ay", "score_chosen", "score_best", "gap_points", "gap_xg", "q_chosen", "q_best", "score_chosen_model", "gap_points_model", "q_chosen_model", "counter_assumed", "best_kind"]].copy()
    out["clock"] = out.minute.astype(int).astype(str).str.zfill(2) + ":" + out.second.astype(int).astype(str).str.zfill(2); out.drop(columns=["minute", "second"]).round(4).to_csv(f"{D}/decision_scores_test.csv", index=False); print(f"\nwrote decision_scores_test.csv with a stratified random sample of {len(out)} of {len(T)} test decisions (seed 0; no player names)")
