"""Pooled leave-one-game-out (LOGO) validation of the geometric feasibility score as a probability estimate.

    python -m src.pooled_validation --out reports/pooled_validation_results.txt --figdir reports

Three folds (train on two games pooled, test on the third) x scenarios A and B (kept separate). Features, labels and
eligibility are exactly those of ``pass_completion_features`` (unchanged). Everything fitted (winsorising/scaling, ridge,
logistic, Platt, base rate) uses the TRAINING games only; the held-out game is scored once.

Deployable models (the only ones whose held-out numbers are claims):
  BASE   training-pool completion rate
  M2     the existing single-feature-set baseline: ridge logistic + Platt on block-CV out-of-fold predictions
         (B: margin_static; A: the two option-set aggregates, A has no single score)
  M3     plain (unpenalised) logistic on ONE score: B margin_static, A best_margin (a-priori choice, the continuous analogue)
  M3x    M3 + Platt fitted on CROSS-GAME out-of-fold predictions inside the training pool (model on game a -> predict game b)
DIAGNOSTICS ONLY (use held-out labels; never deployable, never reported as predictions): oracle intercept shift, oracle
slope+intercept recalibration, common-slope/game-intercept fit, per-game slopes. They separate "different base rates" from
"unstable score-outcome relationship".
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
from src.pass_completion_features import build_features, exclusion_counts
from src.pass_dataset import DecisionContract

GAMES = (1, 2, 3)
SCORE = {"A": "best_margin", "B": "margin_static"}      # single score for M3 (a priori)
N_BOOT = 1000
N_CAL_BOOT = 300


# ---------------------------------------------------------------- models
class SimpleLogit:
    """Unpenalised logistic regression on a winsorised+standardised feature set; optional Platt layer. Train-only fit."""

    def __init__(self, features: list[str]):
        self.features, self.platt = features, None

    def fit(self, tr: pd.DataFrame) -> "SimpleLogit":
        X = tr[self.features].to_numpy(float)
        self.prep = pc.Prep().fit(X)
        self.w, self.b = pc.fit_logistic(self.prep.transform(X), tr["y_completed"].to_numpy(float), 1e-6)
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return pc.sigmoid(self.prep.transform(df[self.features].to_numpy(float)) @ self.w + self.b)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_raw(df)
        return p if self.platt is None else pc.sigmoid(self.platt[0] + self.platt[1] * pe.logit(p))


def crossgame_oof(tables: list[pd.DataFrame], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Cross-game out-of-fold predictions among the TRAINING games: each game is predicted by a model fitted on the others only."""
    ps, ys = [], []
    for k, t in enumerate(tables):
        m = SimpleLogit(features).fit(pd.concat([x for j, x in enumerate(tables) if j != k], ignore_index=True))
        ps.append(m.predict_raw(t))
        ys.append(t["y_completed"].to_numpy(float))
    return np.concatenate(ps), np.concatenate(ys)


def fit_crossgame_platt(tables: list[pd.DataFrame], features: list[str]) -> SimpleLogit:
    """M3x: Platt calibrator from ``crossgame_oof`` (training games only), then the model refit on the full pool."""
    p, y = crossgame_oof(tables, features)
    w, b = pc.fit_logistic(pe.logit(p)[:, None], y, 1e-6)
    m = SimpleLogit(features).fit(pd.concat(tables, ignore_index=True))
    m.platt = (float(b), float(w[0]))
    return m


def fit_deployable(sc: str, tables: list[pd.DataFrame]) -> dict:
    """All deployable predictors from the pooled TRAINING tables only."""
    cfg = pc.SCENARIO[sc]
    tr = pd.concat(tables, ignore_index=True)
    f2 = cfg["sets"][cfg["base2"]]
    _, l2s, _ = pc.select_feature_set(tr, {cfg["base2"]: f2})
    return {"BASE": pc.fit_model(tr, "Base rate", [], "base", None),
            "M2": pc.fit_model(tr, "M2", f2, "logit", l2s[cfg["base2"]]),
            "M3": SimpleLogit([SCORE[sc]]).fit(tr),
            "M3x": fit_crossgame_platt(tables, [SCORE[sc]]),
            "train_rate": float(tr["y_completed"].mean())}


# ---------------------------------------------------------------- diagnostics (held-out labels: NEVER deployable)
def oracle_shift(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """DIAGNOSTIC: shift logit(p) by a constant delta (slope fixed at 1) so that mean(p') equals the held-out completion rate
    (bisection). Returns (shifted probabilities, delta); delta is the base-rate offset on the logit scale."""
    z, target = pe.logit(p), float(np.mean(y))
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.mean(pc.sigmoid(z + mid)) < target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return pc.sigmoid(z + d), float(d)


def oracle_recalibrate(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """DIAGNOSTIC: refit slope and intercept on the held-out labels (an upper bound for any recalibration)."""
    w, b = pc.fit_logistic(pe.logit(p)[:, None], np.asarray(y, float), 1e-6)
    return pc.sigmoid(b + w[0] * pe.logit(p))


def calib_slope_intercept(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope/intercept of logit P(intercepted) (ideal 1 / 0)."""
    c = pe.calibration_assessment(p, y, pc.fit_logistic)
    return c["slope"], c["intercept"]


def calib_ci(p, y, idx: np.ndarray) -> tuple[tuple, tuple]:
    sl, ic = [], []
    for i in idx[:N_CAL_BOOT]:
        s, c = calib_slope_intercept(np.asarray(p)[i], np.asarray(y)[i])
        if np.isfinite(s):
            sl.append(s); ic.append(c)
    pc_ = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return pc_(sl), pc_(ic)


def interception_rate_table(tables: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, float]:
    """Interception rate per game with Wilson 95% CI, plus the chi-square statistic for equal rates (df = games-1;
    for 3 games p = exp(-chi2/2) exactly)."""
    rows, n_tot, k_tot = [], 0, 0
    for g, t in tables.items():
        n, k = len(t), int((1 - t["y_completed"]).sum())
        z = 1.96
        centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append((f"G{g}", n, k, k / n, centre - half, centre + half))
        n_tot += n; k_tot += k
    p0 = k_tot / n_tot
    chi2 = sum((k - n * p0) ** 2 / (n * p0 * (1 - p0)) for _, n, k, *_ in rows)
    return pd.DataFrame(rows, columns=["game", "n", "intercepted", "rate", "wilson_lo", "wilson_hi"]), float(chi2)


def per_game_relationship(tables: dict[int, pd.DataFrame], feat: str, n_boot: int = N_CAL_BOOT) -> pd.DataFrame:
    """DESCRIPTIVE (uses all three games): per-game logistic of completion on the standardised score (common scaling),
    with bootstrap CIs, plus the common-slope / game-specific-intercept fit."""
    allt = pd.concat([t.assign(_g=g) for g, t in tables.items()], ignore_index=True)
    prep = pc.Prep().fit(allt[[feat]].to_numpy(float))
    rng = np.random.default_rng(0)
    rows = []
    for g, t in tables.items():
        z, y = prep.transform(t[[feat]].to_numpy(float)), t["y_completed"].to_numpy(float)
        w, b = pc.fit_logistic(z, y, 1e-6)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            ww, bb = pc.fit_logistic(z[i], y[i], 1e-6)
            bs.append((ww[0], bb))
        bs = np.array(bs)
        rows.append((f"G{g}", len(y), float(w[0]), *np.percentile(bs[:, 0], [2.5, 97.5]), float(b), *np.percentile(bs[:, 1], [2.5, 97.5]),
                     pe.roc_auc(t[feat].to_numpy(float) * -1, y.astype(int))))
    out = pd.DataFrame(rows, columns=["game", "n", "slope_per_SD", "slope_lo", "slope_hi", "intercept", "int_lo", "int_hi", "AUC_score"])
    z = prep.transform(allt[[feat]].to_numpy(float))
    D = np.column_stack([z[:, 0]] + [(allt["_g"] == g).to_numpy(float) for g in list(tables)[1:]])
    w, b = pc.fit_logistic(D, allt["y_completed"].to_numpy(float), 1e-6)
    out.attrs["common"] = {"slope": float(w[0]), "intercepts": {f"G{g}": float(b + (w[1 + i - 1] if i else 0.0)) for i, g in enumerate(tables)}}
    return out


# ---------------------------------------------------------------- experiment
def run_fold(sc: str, tables: dict[int, pd.DataFrame], test_game: int) -> dict:
    train_games = [g for g in tables if g != test_game]
    tr_tables = [tables[g] for g in train_games]
    te = tables[test_game]
    y = te["y_completed"].to_numpy()
    fit = fit_deployable(sc, tr_tables)
    P = {"BASE": fit["BASE"].predict(te), "M2": fit["M2"].predict(te), "M3": fit["M3"].predict(te), "M3x": fit["M3x"].predict(te)}
    single = {g: SimpleLogit([SCORE[sc]]).fit(tables[g]).predict(te) for g in train_games}           # single-game training, same model as M3
    shifted, delta = oracle_shift(P["M3"], y)
    diag = {"oracle_intercept": shifted, "delta": delta, "oracle_recal": oracle_recalibrate(P["M3"], y)}
    return {"sc": sc, "test": test_game, "train": train_games, "te": te, "y": y, "fit": fit, "P": P, "single": single, "diag": diag,
            "idx": boot_indices(len(te), N_BOOT), "train_rates": {g: float((1 - tables[g]["y_completed"]).mean()) for g in train_games},
            "n_train": sum(len(t) for t in tr_tables), "n_int_train": int(sum((1 - t["y_completed"]).sum() for t in tr_tables))}


def _ci(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def report_fold(r: dict, say) -> None:
    y, idx, P = r["y"], r["idx"], r["P"]
    n_int = int((1 - y).sum())
    say(f"\n--- {r['sc']}: train G{r['train'][0]}+G{r['train'][1]} (n={r['n_train']}, {r['n_int_train']} intercepted; interception rate per training game "
        + ", ".join(f"G{g} {v:.3f}" for g, v in r["train_rates"].items())
        + f"; pooled {1 - r['fit']['train_rate']:.3f}) -> test G{r['test']} (n={len(y)}, {n_int} intercepted / {int(y.sum())} completed; rate {n_int / len(y):.3f}) ---")
    base = P["BASE"]
    names = {"BASE": "base rate (pooled train)", "M2": "M2 existing baseline (ridge+Platt)", "M3": "M3 logistic on single score", "M3x": "M3x + cross-game Platt"}
    say(f"{'model':<36}{'ROC-AUC':>24}{'PR-AUC':>24}{'dBrier vs base':>30}{'dLogLoss vs base':>30}{'cal.slope':>20}{'cal.intercept':>20}")
    for k, nm in names.items():
        p = P[k]
        if k == "BASE":
            say(f"{nm:<36}{'0.500 (const.)':>24}{'(no-skill ' + format(n_int / len(y), '.3f') + ')':>24}{'Brier ' + format(pe.brier(p, y), '.4f'):>30}{'log-loss ' + format(pe.log_loss(p, y), '.4f'):>30}{'-':>20}{'-':>20}")
            continue
        a, pr = pe.metric_ci("roc_auc", p, y, idx), pe.metric_ci("pr_auc_intercepted", p, y, idx)
        db, dl = pe.paired_metric_diff("brier", p, base, y, idx), pe.paired_metric_diff("log_loss", p, base, y, idx)
        sl, ic = calib_slope_intercept(p, y)
        (slo, shi), (ilo, ihi) = calib_ci(p, y, idx)
        f = lambda d: f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{'*' if d['significant'] else ' '}"
        say(f"{nm:<36}{_ci(a):>24}{pr[0]:>24.3f}{f(db):>30}{f(dl):>30}{f'{sl:.2f} [{slo:.2f},{shi:.2f}]':>20}{f'{ic:+.2f} [{ilo:+.2f},{ihi:+.2f}]':>20}")
    say("  separate single-game training of M3 on the same held-out samples (does pooling help?):")
    for g, p in r["single"].items():
        sl, ic = calib_slope_intercept(p, y)
        say(f"    train G{g} only: ROC-AUC {pe.roc_auc(p, y):.3f}  Brier {pe.brier(p, y):.4f} (base {pe.brier(base, y):.4f})  log-loss {pe.log_loss(p, y):.4f} (base {pe.log_loss(base, y):.4f})  cal. slope {sl:.2f} intercept {ic:+.2f}")
    p = P["M3"]
    say(f"    pooled M3     : ROC-AUC {pe.roc_auc(p, y):.3f}  Brier {pe.brier(p, y):.4f}  log-loss {pe.log_loss(p, y):.4f}  cal. slope {calib_slope_intercept(p, y)[0]:.2f} intercept {calib_slope_intercept(p, y)[1]:+.2f}")
    say("  DIAGNOSTIC ONLY (uses held-out labels; not deployable): how much of the probability error is the base rate vs the relationship?")
    say(f"    base-rate offset needed on the logit scale (slope fixed at 1): delta = {r['diag']['delta']:+.2f}; share of the log-loss gap to full recalibration recovered by that intercept shift alone: {intercept_share(r):.0%}")
    ll0, b0 = pe.log_loss(base, y), pe.brier(base, y)
    for nm, q in (("M3 as deployed", P["M3"]), ("M3 + oracle intercept shift", r["diag"]["oracle_intercept"]), ("M3 + oracle slope&intercept", r["diag"]["oracle_recal"])):
        say(f"    {nm:<30} Brier {pe.brier(q, y):.4f} (dBase {pe.brier(q, y) - b0:+.4f})  log-loss {pe.log_loss(q, y):.4f} (dBase {pe.log_loss(q, y) - ll0:+.4f})  mean predicted completion {q.mean():.3f} vs observed {y.mean():.3f}")


def intercept_share(r: dict) -> float:
    """DIAGNOSTIC: share of the log-loss gap between M3-as-deployed and the oracle slope+intercept recalibration that an
    intercept (base-rate) shift alone recovers. NaN if the gap is negligible."""
    y = r["y"]
    dep, shift, full = (pe.log_loss(q, y) for q in (r["P"]["M3"], r["diag"]["oracle_intercept"], r["diag"]["oracle_recal"]))
    return (dep - shift) / (dep - full) if dep - full > 1e-4 else float("nan")


def summary_table(folds: dict) -> pd.DataFrame:
    rows = []
    for g, r in folds.items():
        y, P, base = r["y"], r["P"], r["P"]["BASE"]
        sl, _ = calib_slope_intercept(P["M3"], y)
        single_b = np.mean([pe.brier(p, y) for p in r["single"].values()])
        single_l = np.mean([pe.log_loss(p, y) for p in r["single"].values()])
        rows.append((f"test G{g}", 1 - r["fit"]["train_rate"], 1 - y.mean(), 1 - P["M3"].mean(), pe.roc_auc(P["M3"], y),
                     pe.brier(P["M3"], y) - pe.brier(base, y), pe.log_loss(P["M3"], y) - pe.log_loss(base, y),
                     pe.brier(P["M2"], y) - pe.brier(base, y), pe.brier(P["M3x"], y) - pe.brier(base, y),
                     sl, r["diag"]["delta"], intercept_share(r),
                     pe.brier(P["M3"], y) - single_b, pe.log_loss(P["M3"], y) - single_l))
    return pd.DataFrame(rows, columns=["fold", "train_int_rate", "test_int_rate", "mean_pred_int", "AUC_M3", "dBrier_M3", "dLL_M3", "dBrier_M2",
                                       "dBrier_M3x", "cal_slope_M3", "base_rate_offset_delta", "share_LL_gap_by_intercept", "pooled_minus_single_Brier",
                                       "pooled_minus_single_LL"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    tol = cfg["sync"]["tolerance_s"]
    contract, params = DecisionContract.from_config(cfg), FeasibilityParams.from_config(cfg)
    L: list[str] = []
    say = L.append
    loaded = {g: _load(cfg, g) for g in GAMES}
    fs = {g: {sc: build_features(td, ev, sc, tol, contract, params) for sc in "AB"} for g, (td, ev) in loaded.items()}
    say("Pooled leave-one-game-out validation. Games 1,2,3 as supplied (see cross_game_validation.md for the Game 3 identity-swap caveat).")
    say(f"contract {contract}\nfeasibility params {params}\nbootstrap {N_BOOT} resamples over held-out decisions (paired); '*' = 95% CI excludes 0; calibration CIs {N_CAL_BOOT} resamples")
    folds = {}
    for sc in "AB":
        tables = {g: fs[g][sc].table for g in GAMES}
        title = ("A: DECISION TIME (target-free), prediction frame = release; M3 score = best_margin" if sc == "A"
                 else "B: EARLY FLIGHT (identified target), prediction frame = release + 6 frames, NOT decision time; M3 score = margin_static")
        say(f"\n{'=' * 100}\nSCENARIO {title}\n{'=' * 100}")
        say("-- samples and exclusion rates (first failing check) per game")
        rows = []
        for g in GAMES:
            ex = exclusion_counts(fs[g][sc]); tot = ex.sum()
            rows.append((f"G{g}", len(tables[g]), int((1 - tables[g].y_completed).sum()), int(tables[g].y_completed.sum()),
                         1 - ex.loc["KEPT", "completed"] / tot["completed"], 1 - ex.loc["KEPT", "intercepted"] / tot["intercepted"]))
        say(pd.DataFrame(rows, columns=["game", "n", "intercepted", "completed", "excl_rate_completed", "excl_rate_intercepted"]).round(3).to_string(index=False))
        rt, chi2 = interception_rate_table(tables)
        say(f"-- BASE-RATE VARIATION: interception rate per game (Wilson 95% CI); chi-square for equal rates = {chi2:.2f} on 2 df, p = {np.exp(-chi2 / 2):.4f}")
        say(rt.round(3).to_string(index=False))
        say("(the chi-square treats passes as independent; passes within a match are not, so p-values are optimistic)")
        folds[sc] = {g: run_fold(sc, tables, g) for g in GAMES}
        for g in GAMES:
            report_fold(folds[sc][g], say)
        say("\n-- CROSS-FOLD SUMMARY (deployable M3 unless stated; d* = model minus pooled-training base-rate predictor, negative = better; dB/dL pooled-minus-single = pooled M3 minus mean of the two single-game-trained M3)")
        say(summary_table(folds[sc]).round(4).to_string(index=False))
        say(f"\n-- SCORE-OUTCOME RELATIONSHIP PER GAME (DESCRIPTIVE, all three games, common scaling; slope = change in logit P(completed) per +1 SD of {SCORE[sc]}); AUC = ranking of intercepted by the score")
        rel = per_game_relationship(tables, SCORE[sc])
        say(rel.round(3).to_string(index=False))
        c = rel.attrs["common"]
        say(f"common slope {c['slope']:.3f} with game-specific intercepts " + ", ".join(f"{k} {v:+.3f}" for k, v in c["intercepts"].items())
            + "  (diagnostic fit on all three games; not a prediction)")
    if a.figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fd = Path(a.figdir); fd.mkdir(parents=True, exist_ok=True)
        for sc in "AB":
            fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
            for ax, g in zip(axs, GAMES):
                r = folds[sc][g]
                top = 0.05
                for k, lab in (("M3", "M3 raw"), ("M3x", "M3x cross-game Platt"), ("M2", "M2 ridge+Platt")):
                    t = pe.calibration_assessment(r["P"][k], r["y"], pc.fit_logistic)["table"]
                    if len(t):
                        top = max(top, float(t.iloc[:, 1].max()), float(t["observed_rate"].max()))
                        ax.plot(t.iloc[:, 1], t["observed_rate"], "o-", ms=3, label=lab)
                ax.plot([0, 1.15 * top], [0, 1.15 * top], "k--", lw=1)
                ax.set_xlim(0, 1.15 * top); ax.set_ylim(0, 1.15 * top)
                ax.set_title(f"Scenario {sc}: train others -> test G{g}", fontsize=9)
                ax.set_xlabel("mean predicted P(intercepted)"); ax.set_ylabel("observed rate")
                ax.legend(fontsize=7)
            fig.savefig(fd / f"pooled_reliability_{sc}.png", dpi=105, bbox_inches="tight")
            plt.close(fig)
            say(f"figure: {fd / f'pooled_reliability_{sc}.png'}")
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
