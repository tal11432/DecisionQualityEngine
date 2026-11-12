"""Cross-game validation of pass-completion scenarios A (decision time) and B (early flight).

    python -m src.cross_game --out reports/cross_game_results.txt --figdir reports

For every ordered pair (train game, test game) of games 1, 2, 3 and each scenario SEPARATELY:
scaling, ridge strength, feature-set selection (1-SE rule), boosting rounds, Platt calibration are fitted on the
TRAINING game only (``pass_completion.fit_model`` / ``select_feature_set``); the test game is scored once. No threshold,
calibration or feature selection uses test data. Feature definitions are those of ``pass_completion_features``.
A validation gate runs first; evaluation is skipped if any game fails it. Scenario A and B samples are different
populations and are never compared with each other; within a test game all models use identical samples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import pass_completion as pc
from src import pass_completion_evaluation as pe
from src.analytics.feasibility import FeasibilityParams
from src.cli import _load
from src.config import load_config
from src.evaluation import boot_indices
from src.ingestion.epts import apply_identity_swaps, detect_identity_swaps
from src.pass_completion_features import build_features, exclusion_counts, label_events
from src.pass_dataset import DecisionContract
from src.processing.sync import synchronize_events
from src.processing.validation import validate_tracking

GAMES = (1, 2, 3)
N_BOOT = 1000
MIN_INTERCEPTED_FOR_EVAL = 50      # gate: a game needs at least this many labelled interceptions


def gate(cfg: dict, loaded: dict, say) -> bool:
    """Validation + label availability per game. Returns False if any game cannot be compared reliably."""
    ok_all = True
    say("== VALIDATION GATE (tracking validation, event synchronization, label availability) ==")
    for g, (td, ev) in loaded.items():
        rep = validate_tracking(td, cfg)
        res = synchronize_events(ev, td.frames, cfg["sync"]["tolerance_s"])
        lab = label_events(ev)
        n_c, n_i = int((lab.outcome == "completed").sum()), int((lab.outcome == "intercepted").sum())
        sync_ok = len(res.matched) / max(len(ev), 1)
        ok = rep.ok and sync_ok >= 0.99 and n_i >= MIN_INTERCEPTED_FOR_EVAL and n_c > 0
        ok_all &= ok
        warn = [c.name for c in rep.checks if c.status == "WARN"]
        say(f"G{g}: {'PASS' if ok else 'FAIL'}  source={td.source}  frames={len(td.frames)}  players={len(td.players)}  validation={'PASS' if rep.ok else 'FAIL'}"
            f" (warnings: {', '.join(warn) or 'none'})  events synced {len(res.matched)}/{len(ev)}  labels: {n_c} completed / {n_i} intercepted")
        say(f"     timestamps: {rep.get('timestamps').detail}")
        for sw in detect_identity_swaps(td, ev):
            say(f"     WARNING identity swap: {sw['team']} period {sw['period']}: tracking slots {sw['a']} and {sw['b']} look exchanged vs events "
                f"({sw['rate_a_at_b']:.0%} of {sw['n_a']} / {sw['rate_b_at_a']:.0%} of {sw['n_b']} events have the ball at the other player). "
                f"Default: data kept as supplied (those passers fail the possession filter -> excluded, non-randomly); --fix-identity applies the swap.")
    return bool(ok_all)


def _fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def _sig(d):
    return "*" if d["significant"] else " "


def maybe_fix_identity(loaded: dict, fix: bool) -> dict:
    """Opt-in sensitivity repair: apply detected tracking-identity swaps. Default (fix=False) returns the data untouched."""
    if not fix:
        return loaded
    return {g: ((apply_identity_swaps(td, sw) if (sw := detect_identity_swaps(td, ev)) else td), ev) for g, (td, ev) in loaded.items()}


def fit_train(sc: str, tr: pd.DataFrame) -> dict:
    """Everything fitted from the TRAINING table only: feature-set choice, ridge, boosting rounds, scaling, calibration."""
    cfg, sets = pc.SCENARIO[sc], pc.SCENARIO[sc]["sets"]
    chosen, l2s, _ = pc.select_feature_set(tr, sets)
    models = {"Base rate": pc.fit_model(tr, "Base rate", [], "base", None)}
    models[cfg["base1_label"]] = pc.fit_model(tr, cfg["base1_label"], sets[cfg["base1"]], "logit", l2s[cfg["base1"]])
    models[cfg["base2_label"]] = pc.fit_model(tr, cfg["base2_label"], sets[cfg["base2"]], "logit", l2s[cfg["base2"]])
    pname = f"primary logistic [{chosen}]"
    models[pname] = pc.fit_model(tr, pname, sets[chosen], "logit", l2s[chosen])
    rounds = pc.select_rounds(tr, sets[cfg["full"]])
    gname = f"boosted trees [{cfg['full']}, {rounds} rounds]"
    models[gname] = pc.fit_model(tr, gname, sets[cfg["full"]], "gbt", rounds)
    return {"models": models, "pname": pname, "gname": gname, "chosen": chosen, "train_rate": float(tr.y_completed.mean())}


def evaluate_run(sc: str, fit: dict, te: pd.DataFrame) -> dict:
    cfg = pc.SCENARIO[sc]
    y = te["y_completed"].to_numpy()
    idx = boot_indices(len(te), N_BOOT)
    P = {n: m.predict(te) for n, m in fit["models"].items()}
    names = {"base": "Base rate", "b1": cfg["base1_label"], "b2": cfg["base2_label"], "prim": fit["pname"], "gbt": fit["gname"]}
    out = {"n": len(te), "n_int": int((1 - y).sum()), "rate": float(y.mean()), "idx": idx, "P": P, "names": names, "y": y}
    out["metrics"] = {k: {m: pe.metric_ci(m, P[n], y, idx) for m in ("roc_auc", "pr_auc_intercepted", "brier", "log_loss")}
                      for k, n in names.items() if k != "base"}
    out["base"] = {m: pe.metric_ci(m, P[names["base"]], y, idx)[0] for m in ("brier", "log_loss")}
    out["vs_base"] = {k: {m: pe.paired_metric_diff(m, P[names[k]], P[names["base"]], y, idx) for m in ("brier", "log_loss")} for k in ("b1", "b2", "prim", "gbt")}
    pd_ = lambda a, b, m: pe.paired_metric_diff(m, P[names[a]], P[names[b]], y, idx)
    out["feas_vs_dist"] = {m: pd_("b2", "b1", m) for m in ("roc_auc", "log_loss")}
    out["gbt_vs_prim"] = {m: pd_("gbt", "prim", m) for m in ("roc_auc", "log_loss")}
    out["prim_vs_b2"] = {m: pd_("prim", "b2", m) for m in ("roc_auc", "log_loss")}
    out["calib"] = {k: pe.calibration_assessment(P[names[k]], y, pc.fit_logistic) for k in ("prim", "gbt")}
    return out


def verdict(vb: dict, roc: tuple) -> str:
    """Pre-specified: both paired CIs (Brier, log-loss vs training base rate) exclude 0 on the favourable side = 'yes';
    exactly one = 'partial'; none = 'no'. Ranking evidence = ROC-AUC CI excludes 0.5."""
    fav = [vb[m]["significant"] and vb[m]["diff"] < 0 for m in ("brier", "log_loss")]
    prob = "yes" if all(fav) else "partial" if any(fav) else "no"
    return f"prob.skill:{prob}, ranking:{'yes' if roc[1] > 0.5 else 'no'}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--games", default="1,2,3")
    ap.add_argument("--fix-identity", action="store_true",
                    help="SENSITIVITY: apply detected tracking-identity swaps (default: data as supplied)")
    a = ap.parse_args(argv)
    games = tuple(int(g) for g in a.games.split(","))
    cfg = load_config()
    tol = cfg["sync"]["tolerance_s"]
    contract, params = DecisionContract.from_config(cfg), FeasibilityParams.from_config(cfg)
    L: list[str] = []
    say = L.append

    def finish(code: int) -> int:
        text = "\n".join(L)
        print(text)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text + "\n")
        return code

    loaded = {g: _load(cfg, g) for g in games}
    if not gate(cfg, loaded, say):
        say("\nGATE FAILED: at least one game cannot be compared reliably. Model evaluation NOT run.")
        return finish(1)
    say("gate passed for all games\n")
    if a.fix_identity:
        loaded = maybe_fix_identity(loaded, True)
        say("SENSITIVITY RUN: detected identity swaps were APPLIED (see gate warnings); sources: " + ", ".join(f"G{g} {td.source}" for g, (td, _) in loaded.items()) + "\n")
    say(f"contract {contract}\nfeasibility params {params}\nbootstrap {N_BOOT} resamples (paired, shared per test game); '*' = 95% CI excludes 0")

    fs = {g: {sc: build_features(td, ev, sc, tol, contract, params) for sc in "AB"} for g, (td, ev) in loaded.items()}
    results: dict = {}
    for sc in "AB":
        title = ("A: DECISION-TIME (target-free), prediction frame = ball release" if sc == "A"
                 else "B: EARLY-FLIGHT (identified target), prediction frame = release + 6 frames. NOT decision-time")
        say(f"\n{'=' * 90}\nSCENARIO {title}\n{'=' * 90}")
        rows = []
        for g in games:
            ex, t = exclusion_counts(fs[g][sc]), fs[g][sc].table
            rows.append((f"G{g}", len(t), int((1 - t.y_completed).sum()), int(t.y_completed.sum()), t.y_completed.mean(),
                         1 - ex.loc["KEPT", "completed"] / ex.sum()["completed"], 1 - ex.loc["KEPT", "intercepted"] / ex.sum()["intercepted"],
                         len(fs[g][sc].unknown)))
        say("-- samples per game")
        say(pd.DataFrame(rows, columns=["game", "n", "intercepted", "completed", "completion_rate", "excl_rate_completed",
                                        "excl_rate_intercepted", "unknown_outcome_events"]).round(3).to_string(index=False))

        fits = {g: fit_train(sc, fs[g][sc].table) for g in games}
        say("-- feature set selected on each TRAINING game (1-SE rule): " + "; ".join(f"G{g}: {fits[g]['chosen']}" for g in games))

        runs = {(a_, b_): evaluate_run(sc, fits[a_], fs[b_][sc].table) for a_ in games for b_ in games if a_ != b_}
        results[sc] = runs
        say(f"\n-- RUNS (train -> test). Brier/log-loss differences are model minus the TRAINING-game base-rate predictor on the same test samples (negative = better)")
        for (tr_g, te_g), r in runs.items():
            m = r["metrics"]
            say(f"\nG{tr_g} -> G{te_g}: n={r['n']} ({r['n_int']} intercepted; test completion rate {r['rate']:.3f} vs train {fits[tr_g]['train_rate']:.3f})")
            for k, label in (("b1", "distance/location baseline"), ("b2", "feasibility baseline"), ("prim", "PRIMARY logistic"), ("gbt", "boosted trees (exploratory)")):
                vb = r["vs_base"][k]
                say(f"   {label:<28} ROC-AUC {_fmt(m[k]['roc_auc'])}  PR-AUC {m[k]['pr_auc_intercepted'][0]:.3f}  "
                    f"dBrier {vb['brier']['diff']:+.4f} [{vb['brier']['lo']:+.4f}, {vb['brier']['hi']:+.4f}]{_sig(vb['brier'])}  "
                    f"dLogLoss {vb['log_loss']['diff']:+.4f} [{vb['log_loss']['lo']:+.4f}, {vb['log_loss']['hi']:+.4f}]{_sig(vb['log_loss'])}  -> {verdict(vb, m[k]['roc_auc'])}")
            say(f"   (no-skill PR-AUC = {r['n_int'] / r['n']:.3f}; base-rate predictor Brier {r['base']['brier']:.4f}, log-loss {r['base']['log_loss']:.4f})")

        say("\n-- SUMMARY MATRICES: ROC-AUC (train rows x test columns; diagonal not evaluated)")
        for k, label in (("b1", "distance/location baseline"), ("b2", "feasibility baseline"), ("prim", "primary logistic"), ("gbt", "boosted trees")):
            mat = pd.DataFrame({f"test G{b}": {f"train G{a_}": (runs[(a_, b)]["metrics"][k]["roc_auc"][0] if a_ != b else np.nan) for a_ in games} for b in games})
            say(f"{label}:\n{mat.round(3).to_string()}")

        say("\n-- PAIRED COMPARISONS ON IDENTICAL TEST SAMPLES (first minus second; '*' = CI excludes 0)")
        say(f"{'run':<9}{'feasibility - ' + ('distance' if sc == 'B' else 'location') + ' (AUC)':<46}{'(log-loss)':<34}{'boosted - logistic (AUC)':<34}{'(log-loss)':<34}")
        for (tr_g, te_g), r in runs.items():
            f_, g_ = r["feas_vs_dist"], r["gbt_vs_prim"]
            c = lambda d: f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{_sig(d)}"
            say(f"G{tr_g}->G{te_g}  {c(f_['roc_auc']):<44}{c(f_['log_loss']):<34}{c(g_['roc_auc']):<34}{c(g_['log_loss']):<34}")

        say("\n-- PROBABILITY QUALITY: calibration slope / intercept of logit P(intercepted) on the test game (diagnostic, never applied; ideal 1 / 0)")
        for (tr_g, te_g), r in runs.items():
            cp, cg = r["calib"]["prim"], r["calib"]["gbt"]
            tab = cp["table"]
            span = f"predicted P(int) {tab.iloc[0, 1]:.3f}-{tab.iloc[-1, 1]:.3f} vs observed {tab['observed_rate'].iloc[0]:.3f}-{tab['observed_rate'].iloc[-1]:.3f} ({cp['n_bins']} bins)" if len(tab) else "too few events for bins"
            say(f"G{tr_g}->G{te_g}: primary slope {cp['slope']:.2f} int {cp['intercept']:+.2f} | boosted slope {cg['slope']:.2f} int {cg['intercept']:+.2f} | {span}")

        say("\n-- DOES A MODEL BEAT THE TRAINING BASE RATE ON BOTH Brier AND log-loss (6 runs)?")
        for k, label in (("b2", "feasibility baseline"), ("prim", "primary logistic"), ("gbt", "boosted trees")):
            v = [verdict(r["vs_base"][k], r["metrics"][k]["roc_auc"]) for r in runs.values()]
            say(f"{label:<22} prob.skill yes {sum('prob.skill:yes' in x for x in v)}, partial {sum('prob.skill:partial' in x for x in v)}, no {sum('prob.skill:no' in x for x in v)}"
                f" | ranking (AUC CI > 0.5) in {sum('ranking:yes' in x for x in v)}/6 runs")
        say("by test game (mean ROC-AUC over the two training games): " + "; ".join(
            f"G{b}: " + ", ".join(f"{k} {np.mean([runs[(a_, b)]['metrics'][k]['roc_auc'][0] for a_ in games if a_ != b]):.3f}" for k in ("b2", "prim", "gbt")) for b in games))
        say("by train game (mean ROC-AUC over the two test games): " + "; ".join(
            f"G{a_}: " + ", ".join(f"{k} {np.mean([runs[(a_, b)]['metrics'][k]['roc_auc'][0] for b in games if b != a_]):.3f}" for k in ("b2", "prim", "gbt")) for a_ in games))

        if a.figdir:
            fd = Path(a.figdir)
            fd.mkdir(parents=True, exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            top = 0.05
            for (tr_g, te_g), r in runs.items():
                t = r["calib"]["prim"]["table"]
                if len(t):
                    top = max(top, float(t.iloc[:, 1].max()), float(t["observed_rate"].max()))
                    ax.plot(t.iloc[:, 1], t["observed_rate"], "o-", ms=3, label=f"G{tr_g}->G{te_g}")
            ax.plot([0, 1.15 * top], [0, 1.15 * top], "k--", lw=1)
            ax.set_xlim(0, 1.15 * top); ax.set_ylim(0, 1.15 * top)
            ax.set_xlabel("mean predicted P(intercepted)"); ax.set_ylabel("observed interception rate")
            ax.set_title(f"Scenario {sc}: primary-model reliability, cross-game", fontsize=9)
            ax.legend(fontsize=7)
            fig.savefig(fd / f"cross_game_reliability_{sc}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            say(f"figure: {fd / f'cross_game_reliability_{sc}.png'}")
    return finish(0)


if __name__ == "__main__":
    raise SystemExit(main())
