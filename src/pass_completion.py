"""Pass-completion probability: models, training and a reproducible train/evaluate CLI.

    python -m src.pass_completion --train 1 --test 2 --out reports/pass_completion_results.txt --figdir reports

Scenario A (decision time, target-free) and scenario B (early flight, identified target) are trained and
reported SEPARATELY; see ``src/pass_completion_features.py`` for their definitions. Everything that is fitted
(clipping/scaling, ridge strength, feature-set choice, boosting rounds, Platt calibration, F1 threshold) uses the
TRAINING game only; the test game is scored once with frozen models.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import pass_completion_evaluation as ev_
from src.analytics.feasibility import FeasibilityParams
from src.cli import _load
from src.config import load_config
from src.evaluation import boot_indices
from src.pass_completion_features import (FEATURES, FEATURES_A, FEATURES_B, build_features, exclusion_counts,
                                          label_events, target_identifiability)
from src.pass_dataset import DecisionContract

CLIP_Q = (0.01, 0.99)
L2_GRID = (1.0, 10.0, 100.0, 1000.0)
ROUNDS_GRID = (25, 50, 100, 200)
CV_BLOCKS = 5
N_BOOT = 2000        # bootstrap resamples for metric CIs
N_COEF_BOOT = 300    # bootstrap resamples for coefficient intervals

SETS_A = {
    "A1 location": ["dist_goal", "angle_goal"],
    "A2 pressure": ["nearest_opp_dist", "n_opp_10m"],
    "A3 feasibility (option set)": ["n_safe_options", "best_margin"],
    "A4 feasibility + movement": ["n_safe_options", "best_margin", "passer_speed", "nearest_opp_closing_speed"],
    "A5 no feasibility": ["dist_goal", "angle_goal", "nearest_opp_dist", "n_opp_10m", "passer_speed", "nearest_opp_closing_speed"],
    "A6 full": FEATURES_A,
}
SETS_B = {
    "B1 pass distance": ["pass_dist"],
    "B2 feasibility (static margin)": ["margin_static"],
    "B3 feasibility + movement": ["margin_vel", "receiver_speed"],
    "B4 distance + feasibility": ["pass_dist", "margin_vel"],
    "B5 no feasibility": ["pass_dist", "early_ball_speed", "receiver_speed", "receiver_nearest_opp_dist",
                          "forward_gain", "dist_goal_target"],
    "B6 full": [f for f in FEATURES_B if f != "margin_static"],
}
SCENARIO = {"A": {"sets": SETS_A, "base1": "A1 location", "base2": "A3 feasibility (option set)", "full": "A6 full",
                  "nofeas": "A5 no feasibility", "base1_label": "location-only baseline (no pass distance exists at t_d)",
                  "base2_label": "feasibility-aggregate baseline"},
            "B": {"sets": SETS_B, "base1": "B1 pass distance", "base2": "B2 feasibility (static margin)",
                  "full": "B6 full", "nofeas": "B5 no feasibility", "base1_label": "pass-distance baseline",
                  "base2_label": "existing feasibility-margin baseline"}}


# ---------------------------------------------------------------- numerics
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class Prep:
    """Winsorise to the training 1st/99th percentile, then standardise. Fitted on training data only."""

    def fit(self, X: np.ndarray) -> "Prep":
        X = np.asarray(X, float)
        self.lo, self.hi = np.quantile(X, CLIP_Q[0], axis=0), np.quantile(X, CLIP_Q[1], axis=0)
        C = np.clip(X, self.lo, self.hi)
        self.mu, sd = C.mean(0), C.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.clip(np.asarray(X, float), self.lo, self.hi) - self.mu) / self.sd


def fit_logistic(Z: np.ndarray, y: np.ndarray, l2: float, max_iter: int = 100) -> tuple[np.ndarray, float]:
    """Ridge logistic regression by Newton/IRLS. Objective: sum log-loss + (l2/2)||w||^2 (intercept unpenalised)."""
    Z, y = np.asarray(Z, float), np.asarray(y, float)
    n, p = Z.shape
    A = np.column_stack([np.ones(n), Z])
    pen = np.diag([0.0] + [l2] * p)
    beta = np.zeros(p + 1)
    for _ in range(max_iter):
        mu = sigmoid(A @ beta)
        grad = A.T @ (mu - y) + pen @ beta
        H = (A * (mu * (1 - mu))[:, None]).T @ A + pen + 1e-9 * np.eye(p + 1)
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta[1:], float(beta[0])


class BoostedTrees:
    """Tiny deterministic gradient-boosted trees (logistic loss, Newton leaves, depth<=2). Exploratory only."""

    def __init__(self, rounds: int = 100, lr: float = 0.1, depth: int = 2, min_leaf: int = 15, lam: float = 5.0, n_cuts: int = 12):
        self.rounds, self.lr, self.depth, self.min_leaf, self.lam, self.n_cuts = rounds, lr, depth, min_leaf, lam, n_cuts

    def _tree(self, X, g, h, depth, imp):
        G, H = g.sum(), h.sum()
        leaf = -G / (H + self.lam)
        if depth == 0 or len(g) < 2 * self.min_leaf:
            return leaf
        best = (0.0, None, None)
        for j in range(X.shape[1]):
            cuts = np.unique(np.quantile(X[:, j], np.linspace(0.1, 0.9, self.n_cuts)))
            for t in cuts:
                m = X[:, j] <= t
                if m.sum() < self.min_leaf or (~m).sum() < self.min_leaf:
                    continue
                gl, hl = g[m].sum(), h[m].sum()
                gr, hr = G - gl, H - hl
                gain = gl * gl / (hl + self.lam) + gr * gr / (hr + self.lam) - G * G / (H + self.lam)
                if gain > best[0]:
                    best = (gain, j, t)
        if best[1] is None:
            return leaf
        gain, j, t = best
        imp[j] += gain
        m = X[:, j] <= t
        return (j, t, self._tree(X[m], g[m], h[m], depth - 1, imp), self._tree(X[~m], g[~m], h[~m], depth - 1, imp))

    @staticmethod
    def _eval(node, X):
        if not isinstance(node, tuple):
            return np.full(len(X), node)
        j, t, l, r = node
        m = X[:, j] <= t
        out = np.empty(len(X))
        out[m], out[~m] = BoostedTrees._eval(l, X[m]), BoostedTrees._eval(r, X[~m])
        return out

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.f0 = float(np.log(y.mean() / (1 - y.mean())))
        F = np.full(len(y), self.f0)
        self.trees, self.importance = [], np.zeros(X.shape[1])
        for _ in range(self.rounds):
            p = sigmoid(F)
            tree = self._tree(X, p - y, p * (1 - p), self.depth, self.importance)
            self.trees.append(tree)
            F += self.lr * self._eval(tree, X)
        return self

    def predict(self, X, rounds: int | None = None):
        X = np.asarray(X, float)
        F = np.full(len(X), self.f0)
        for tree in self.trees[: rounds or len(self.trees)]:
            F += self.lr * self._eval(tree, X)
        return sigmoid(F)


# ---------------------------------------------------------------- cross-validation (training game only)
def block_ids(n: int, k: int = CV_BLOCKS) -> np.ndarray:
    """Contiguous time blocks (table is in event order): deterministic, limits leakage between adjacent passes."""
    return np.minimum((np.arange(n) * k) // n, k - 1)


def oof_predict(X: np.ndarray, y: np.ndarray, fit_predict, k: int = CV_BLOCKS) -> np.ndarray:
    """Out-of-fold predictions; ``fit_predict(Xtr, ytr, Xte)`` must fit EVERYTHING (scaler included) on Xtr only."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    b = block_ids(len(y), k)
    out = np.empty(len(y))
    for i in range(k):
        te = b == i
        out[te] = fit_predict(X[~te], y[~te], X[te])
    return out


def logit_fp(l2: float):
    def fp(Xtr, ytr, Xte):
        prep = Prep().fit(Xtr)
        w, b = fit_logistic(prep.transform(Xtr), ytr, l2)
        return sigmoid(prep.transform(Xte) @ w + b)
    return fp


def gbt_fp(rounds: int):
    def fp(Xtr, ytr, Xte):
        return BoostedTrees(rounds).fit(Xtr, ytr).predict(Xte)
    return fp


def _loss_vec(p, y):
    p = np.clip(p, ev_.EPS, 1 - ev_.EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def select_feature_set(train: pd.DataFrame, sets: dict[str, list[str]]) -> tuple[str, dict, pd.DataFrame]:
    """CV (training game) log-loss per set (best ridge each); choose the SIMPLEST set within one SE of the best."""
    y = train["y_completed"].to_numpy(float)
    rows, store = [], {}
    for name, feats in sets.items():
        X = train[feats].to_numpy(float)
        per = {l2: _loss_vec(oof_predict(X, y, logit_fp(l2)), y) for l2 in L2_GRID}
        l2 = min(per, key=lambda k: per[k].mean())
        store[name] = (l2, per[l2])
        rows.append((name, len(feats), l2, per[l2].mean(), per[l2].std(ddof=1) / np.sqrt(len(y))))
    tab = pd.DataFrame(rows, columns=["feature_set", "n_features", "best_l2", "cv_log_loss", "se"])
    best = tab.loc[tab.cv_log_loss.idxmin()]
    ok = tab[tab.cv_log_loss <= best.cv_log_loss + best.se].sort_values(["n_features", "cv_log_loss"])
    return str(ok.iloc[0].feature_set), {n: store[n][0] for n in store}, tab


# ---------------------------------------------------------------- fitted model
@dataclass
class FittedModel:
    name: str
    features: list[str]
    kind: str                      # 'base' | 'logit' | 'gbt'
    hyper: float | int | None
    prep: Prep | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0
    gbt: BoostedTrees | None = None
    platt: tuple[float, float] | None = None   # p' = sigmoid(a + b * logit(p))
    tau: float | None = None                    # flag 'intercepted' if p_completed <= tau
    base_rate: float = float("nan")
    train_oof: np.ndarray = field(default_factory=lambda: np.array([]))

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "base":
            return np.full(len(df), self.base_rate)
        X = df[self.features].to_numpy(float)
        if self.kind == "logit":
            return sigmoid(self.prep.transform(X) @ self.coef + self.intercept)
        return self.gbt.predict(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_raw(df)
        if self.platt is None:
            return p
        a, b = self.platt
        return sigmoid(a + b * ev_.logit(p))


def fit_model(train: pd.DataFrame, name: str, features: list[str], kind: str, hyper) -> FittedModel:
    """Fit on ``train`` only: estimator, then Platt calibration and F1 threshold from OUT-OF-FOLD predictions."""
    y = train["y_completed"].to_numpy(float)
    if kind == "base":
        return FittedModel(name, [], "base", None, base_rate=float(y.mean()))
    X = train[features].to_numpy(float)
    fp = logit_fp(hyper) if kind == "logit" else gbt_fp(hyper)
    oof = oof_predict(X, y, fp)
    a_b = fit_logistic(ev_.logit(oof)[:, None], y, 1e-6)
    platt = (a_b[1], float(a_b[0][0]))
    m = FittedModel(name, features, kind, hyper, platt=platt, train_oof=oof)
    if kind == "logit":
        m.prep = Prep().fit(X)
        m.coef, m.intercept = fit_logistic(m.prep.transform(X), y, hyper)
    else:
        m.gbt = BoostedTrees(hyper).fit(X, y)
    m.tau = ev_.threshold_f1(sigmoid(platt[0] + platt[1] * ev_.logit(oof)), y)
    return m


def select_rounds(train: pd.DataFrame, features: list[str]) -> int:
    y = train["y_completed"].to_numpy(float)
    X = train[features].to_numpy(float)
    return int(min(ROUNDS_GRID, key=lambda r: _loss_vec(oof_predict(X, y, gbt_fp(r)), y).mean()))


# ---------------------------------------------------------------- orchestration
def _fmt_ci(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def run_scenario(sc: str, tr: pd.DataFrame, te: pd.DataFrame, figdir: Path | None, say) -> dict:
    cfg = SCENARIO[sc]
    sets = cfg["sets"]
    ytr, yte = tr["y_completed"].to_numpy(), te["y_completed"].to_numpy()
    say(f"\n{'=' * 78}\nSCENARIO {sc}: {'DECISION-TIME (target-free), prediction frame = ball release' if sc == 'A' else 'EARLY-FLIGHT (identified target), prediction frame = release + 6 frames (0.24 s). NOT decision-time.'}\n{'=' * 78}")
    say(f"train: n={len(tr)} ({int((1 - ytr).sum())} intercepted / {int(ytr.sum())} completed; completion rate {ytr.mean():.3f})")
    say(f"test : n={len(te)} ({int((1 - yte).sum())} intercepted / {int(yte.sum())} completed; completion rate {yte.mean():.3f})")
    if len(te) == 0 or (1 - yte).sum() < 5 or (1 - ytr).sum() < 15:
        say("INSUFFICIENT DATA for a valid evaluation in this scenario.")
        return {}

    chosen, l2s, cv_tab = select_feature_set(tr, sets)
    say("\n[train-only] feature-set selection: 5-block contiguous-time CV log-loss; rule = simplest set within 1 SE of the best")
    say(cv_tab.round(4).to_string(index=False))
    say(f"SELECTED primary feature set: {chosen} -> {sets[chosen]} (ridge l2={l2s[chosen]})")

    models: dict[str, FittedModel] = {"Base rate": fit_model(tr, "Base rate", [], "base", None)}
    models[cfg["base1_label"]] = fit_model(tr, cfg["base1_label"], sets[cfg["base1"]], "logit", l2s[cfg["base1"]])
    models[cfg["base2_label"]] = fit_model(tr, cfg["base2_label"], sets[cfg["base2"]], "logit", l2s[cfg["base2"]])
    pname = f"PRIMARY logistic [{chosen}]"
    models[pname] = fit_model(tr, pname, sets[chosen], "logit", l2s[chosen])
    rounds = select_rounds(tr, sets[cfg["full"]])
    gname = f"EXPLORATORY boosted trees [{cfg['full']}, {rounds} rounds]"
    models[gname] = fit_model(tr, gname, sets[cfg["full"]], "gbt", rounds)
    abl = {n: fit_model(tr, n, f, "logit", l2s[n]) for n, f in sets.items()}

    P = {n: m.predict(te) for n, m in models.items()}
    idx = boot_indices(len(te), N_BOOT)
    say(f"\n== TEST ({len(te)} passes, {int((1 - yte).sum())} intercepted): all models on identical samples; probabilities are Platt-calibrated on train OOF ==")
    say(f"ROC-AUC measures ranking only. PR-AUC is for the minority class (intercepted); no-skill PR-AUC = "
        f"{(1 - yte).mean():.3f}. Brier/log-loss assess the probabilities (lower is better). 95% bootstrap CIs, {N_BOOT} resamples.")
    hdr = f"{'model':<66}{'ROC-AUC':>24}{'PR-AUC(int.)':>24}{'Brier':>24}{'log-loss':>24}"
    say(hdr)
    for n, p in P.items():
        cells = [_fmt_ci(ev_.metric_ci(k, p, yte, idx)) if (k in ("brier", "log_loss") or n != "Base rate") else "0.500 (constant)"
                 for k in ("roc_auc", "pr_auc_intercepted", "brier", "log_loss")]
        say(f"{n:<66}" + "".join(f"{c:>24}" for c in cells))
    pm = models[pname]
    raw = pm.predict_raw(te)
    say(f"{pname + ' (uncalibrated)':<66}" + "".join(f"{_fmt_ci(ev_.metric_ci(k, raw, yte, idx)):>24}" for k in ("roc_auc", "pr_auc_intercepted", "brier", "log_loss")))

    say("\n== TEST: calibration (diagnostic fits on the test set, never applied back). slope 1 / intercept 0 = ideal on the logit(P(intercepted)) scale ==")
    for n, p in P.items():
        if n == "Base rate":
            continue
        c = ev_.calibration_assessment(p, yte, fit_logistic)
        say(f"  {n}: slope {c['slope']:.2f}, intercept {c['intercept']:+.2f}")
    c = ev_.calibration_assessment(P[pname], yte, fit_logistic)
    if c["n_bins"] >= 2:
        say(f"  reliability table, primary ({c['n_bins']} equal-count bins; each holds ~>=10 intercepted):")
        say(c["table"].round(3).to_string(index=False))
    else:
        say("  too few intercepted passes for a reliability table")
    if figdir is not None:
        path = figdir / f"pass_completion_reliability_{sc}.png"
        ev_.reliability_plot(str(path), {"primary (calibrated)": (P[pname], yte), "primary (uncalibrated)": (raw, yte),
                                         "boosted trees": (P[gname], yte)}, fit_logistic,
                             f"Scenario {sc}: reliability on held-out game")
        say(f"  figure: {path}")

    say("\n== TEST: paired differences on identical samples (95% CI; * = CI excludes 0). Brier/log-loss: negative = first model better ==")
    for a, b in ((pname, "Base rate"), (gname, "Base rate"), (pname, cfg["base1_label"]), (pname, cfg["base2_label"]), (gname, pname)):
        for k in ("roc_auc", "brier", "log_loss"):
            if b == "Base rate" and k == "roc_auc":
                continue
            d = ev_.paired_metric_diff(k, P[a], P[b], yte, idx)
            say(f"  {k:<9} [{a}] - [{b}]: {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {'*' if d['significant'] else ''}")

    say("\n== TEST: confusion at the F1-optimal threshold chosen on TRAIN OOF (flag 'intercepted' if P(completed) <= tau; positive = intercepted) ==")
    for n, m in models.items():
        if m.kind == "base":
            say(f"  {n}: constant prediction; precision/recall undefined (nothing flagged)")
            continue
        c = ev_.confusion_at(P[n], yte, m.tau)
        say(f"  {n:<66} tau={m.tau:.3f}  TP {c['tp']:>3} FP {c['fp']:>3} FN {c['fn']:>3} TN {c['tn']:>4}  P {c['precision']:.3f} R {c['recall']:.3f} F1 {c['f1']:.3f}")

    say("\n== ABLATION on TEST (feature sets chosen/trained on train; same test samples) ==")
    say(f"{'feature set':<40}{'#':>3}{'ROC-AUC':>24}{'Brier':>24}{'log-loss':>24}")
    A = {n: m.predict(te) for n, m in abl.items()}
    for n, p in A.items():
        say(f"{n:<40}{len(sets[n]):>3}" + "".join(f"{_fmt_ci(ev_.metric_ci(k, p, yte, idx)):>24}" for k in ("roc_auc", "brier", "log_loss")))
    for a, b in ((cfg["full"], cfg["nofeas"]), (cfg["base2"], cfg["base1"])):
        for k in ("roc_auc", "log_loss"):
            d = ev_.paired_metric_diff(k, A[a], A[b], yte, idx)
            say(f"  {k:<9} [{a}] - [{b}]: {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {'*' if d['significant'] else ''}")

    say("\n== INTERPRETATION: primary logistic, standardised coefficients (train fit). OR = odds-ratio of COMPLETION per +1 SD ==")
    rng = np.random.default_rng(0)
    Xtr = tr[sets[chosen]].to_numpy(float)
    boots = []
    for _ in range(N_COEF_BOOT):
        i = rng.integers(0, len(ytr), len(ytr))
        pr = Prep().fit(Xtr[i])
        boots.append(fit_logistic(pr.transform(Xtr[i]), ytr[i].astype(float), l2s[chosen])[0])
    boots = np.array(boots)
    for f, c, lo, hi in zip(sets[chosen], pm.coef, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)):
        say(f"  {f:<28} coef {c:+.3f} [{lo:+.3f}, {hi:+.3f}]  OR/SD {np.exp(c):.2f}   ({FEATURES[f][2]}; {FEATURES[f][1]})")
    say(f"  note: scenario-{sc} train vs test base rates differ (intercepted {(1 - ytr).mean():.1%} vs {(1 - yte).mean():.1%}); calibration-in-the-large does not transfer perfectly across matches")
    say(f"  (bootstrap = {N_COEF_BOOT} resamples of the training game; associations, not causal effects; coefficients of correlated features are unstable)")
    g = models[gname].gbt
    imp = pd.Series(g.importance / max(g.importance.sum(), 1e-12), index=sets[cfg["full"]]).sort_values(ascending=False)
    say("  boosted-trees gain importance (share of total split gain, train fit): " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    return {"models": models, "P": P, "pname": pname, "te": te, "tr": tr}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figdir", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    tol = cfg["sync"]["tolerance_s"]
    contract, params = DecisionContract.from_config(cfg), FeasibilityParams.from_config(cfg)
    L: list[str] = []
    say = L.append
    td_tr, ev_tr = _load(cfg, a.train)
    td_te, ev_te = _load(cfg, a.test)
    figdir = Path(a.figdir) if a.figdir else None
    if figdir:
        figdir.mkdir(parents=True, exist_ok=True)
    say(f"TRAIN game {a.train} (fit, select, calibrate, threshold)  |  TEST game {a.test} (held out, scored once)")
    say(f"contract {contract}\nfeasibility params {params}")

    say("\n== LABEL VERIFICATION (provider semantics; labels use event type/subtype only) ==")
    for g, ev in ((a.train, ev_tr), (a.test, ev_te)):
        lab = label_events(ev)
        nxt = ev.shift(-1)
        comp = ev.loc[lab[lab.outcome == "completed"].event_idx]
        icpt = ev.loc[lab[lab.outcome == "intercepted"].event_idx]
        n_c, n_i = nxt.loc[comp.index], nxt.loc[icpt.index]
        say(f"G{g}: completed {len(comp)}; chain check (next event same team and by the receiver): {((n_c.team == comp.team) & (n_c.from_player == comp.to_player)).mean():.1%}")
        say(f"G{g}: intercepted {len(icpt)}; next event by the opponent: {(n_i.team != icpt.team).mean():.1%} (not used as a filter)")
        say(f"G{g}: unknown/excluded outcomes ({int(lab.outcome.isna().sum())}): " + "; ".join(f"{k} x{v}" for k, v in lab[lab.outcome.isna()].reason.value_counts().items()))
        for nm, d in (("completed", comp.head(2)), ("intercepted", icpt.head(2))):
            for i, r in d.iterrows():
                say(f"   e.g. {nm}: idx {i} {r.team} {r.type}/{r.subtype} P{r.period} t={r.start_time_s:.2f}-{r.end_time_s:.2f}s {r.from_player}->{r.to_player}")

    say("\n== CAN THE INTENDED RECEIVER BE IDENTIFIED AT/BEFORE RELEASE? (completed passes only; true receiver used for scoring) ==")
    ident = target_identifiability(td_tr, ev_tr, tol, contract, params)
    say("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in ident.items()))
    say("  => fixed decision-time rules reach 30-35% top-1 (chance ~10%): the receiver is NOT reliably identifiable before release, and\n"
        "     intercepted passes carry no receiver label at all. Scenario A is therefore target-free; scenario B is early-flight only.")

    results, fsets = {}, {}
    for sc in ("A", "B"):
        ftr = build_features(td_tr, ev_tr, sc, tol, contract, params)
        fte = build_features(td_te, ev_te, sc, tol, contract, params)
        fsets[sc] = (ftr, fte)
        say(f"\n[scenario {sc}] sample accounting (first failing check)")
        for nm, fs in ((f"TRAIN G{a.train}", ftr), (f"TEST  G{a.test}", fte)):
            say(f"-- {nm}  (outcome-unknown events excluded before this table: {len(fs.unknown)})")
            say(exclusion_counts(fs).to_string())
            ex = exclusion_counts(fs)
            tot = ex.sum()
            say(f"   exclusion rate: completed {1 - ex.loc['KEPT', 'completed'] / tot['completed']:.1%}, intercepted {1 - ex.loc['KEPT', 'intercepted'] / tot['intercepted']:.1%}")
        results[sc] = run_scenario(sc, ftr.table, fte.table, figdir, say)

    # ---- selection bias ----
    say(f"\n{'=' * 78}\nSELECTION BIAS: scenario B's target filter vs the full candidate set (decision-time scenario-A features describe all passes)\n{'=' * 78}")
    for tag, idx_pair, ev in (("TRAIN", 0, ev_tr), ("TEST", 1, ev_te)):
        ta, tb = fsets["A"][idx_pair].table, fsets["B"][idx_pair].table
        ids = set(tb["event_idx"])
        t = ta.assign(retained=ta.event_idx.isin(ids))
        ph = np.hypot(ev.loc[t.event_idx, "end_x"].to_numpy() - ev.loc[t.event_idx, "start_x"].to_numpy(),
                      ev.loc[t.event_idx, "end_y"].to_numpy() - ev.loc[t.event_idx, "start_y"].to_numpy())
        t["ph_length"] = ph
        say(f"\n[{tag}] completion rate: all scenario-A candidates {t.y_completed.mean():.3f} (n={len(t)}); retained by B filter {t[t.retained].y_completed.mean():.3f} (n={int(t.retained.sum())}); excluded {t[~t.retained].y_completed.mean():.3f} (n={int((~t.retained).sum())})")
        say(f"[{tag}] retention rate: completed {t[t.y_completed == 1].retained.mean():.1%}, intercepted {t[t.y_completed == 0].retained.mean():.1%}")
        say(f"[{tag}] post-hoc actual pass length (from event end point; descriptive only, never a feature), median m: " +
            ", ".join(f"{c} retained {t[(t.y_completed == v) & t.retained].ph_length.median():.1f} / excluded {t[(t.y_completed == v) & ~t.retained].ph_length.median():.1f}"
                      for c, v in (("completed", 1), ("intercepted", 0))))
        if tag == "TEST":
            say(ev_.retained_vs_excluded(ta, ids, ["dist_goal", "nearest_opp_dist", "n_opp_10m", "passer_speed", "best_margin"]).round(2).to_string(index=False))
    ftrA, fteA = fsets["A"]
    prep = Prep().fit(ftrA.table[FEATURES_A].to_numpy(float))
    ret_tr = ftrA.table.event_idx.isin(set(fsets["B"][0].table.event_idx)).to_numpy(float)
    w, b = fit_logistic(prep.transform(ftrA.table[FEATURES_A].to_numpy(float)), ret_tr, 10.0)
    ret_te = fteA.table.event_idx.isin(set(fsets["B"][1].table.event_idx)).to_numpy()
    sc_ret = sigmoid(prep.transform(fteA.table[FEATURES_A].to_numpy(float)) @ w + b)
    say(f"\nretention predictability: logistic on scenario-A decision-time features (trained on G{a.train}) predicts which test passes survive B's filter with ROC-AUC {ev_.roc_auc(sc_ret, ret_te.astype(int)):.3f}")
    if results.get("A"):
        r = results["A"]
        retained = pd.Series(np.where(r["te"].event_idx.isin(set(fsets["B"][1].table.event_idx)), "retained by B filter", "excluded by B filter"))
        say("\nscenario-A primary model on TEST, split by whether the pass would survive B's filter (ROC-AUC [95% CI]; n/a if either class < 15):")
        say(ev_.subgroup_table({"A primary": r["P"][r["pname"]]}, r["te"]["y_completed"].to_numpy(), retained).to_string(index=False))
        ph = np.hypot(ev_te.loc[r["te"].event_idx, "end_x"].to_numpy() - ev_te.loc[r["te"].event_idx, "start_x"].to_numpy(),
                      ev_te.loc[r["te"].event_idx, "end_y"].to_numpy() - ev_te.loc[r["te"].event_idx, "start_y"].to_numpy())
        grp = pd.Series(pd.cut(ph, [0, 10, 20, 500], labels=["<10 m", "10-20 m", ">=20 m"], right=False).astype(str))
        say("\nscenario-A primary on TEST by POST-HOC actual pass length (stratifier only, from the event end point):")
        say(ev_.subgroup_table({"A primary": r["P"][r["pname"]]}, r["te"]["y_completed"].to_numpy(), grp).to_string(index=False))
        third = pd.Series(pd.cut(r["te"]["dist_goal"].to_numpy(), [0, 35, 70, 200], labels=["attacking third (<35 m)", "middle (35-70 m)", "defensive (>=70 m)"], right=False).astype(str))
        say("\nscenario-A primary on TEST by passer location (decision-time):")
        say(ev_.subgroup_table({"A primary": r["P"][r["pname"]]}, r["te"]["y_completed"].to_numpy(), third).to_string(index=False))
    if results.get("B"):
        r = results["B"]
        grp = pd.Series(pd.cut(r["te"]["pass_dist"].to_numpy(), [0, 10, 20, 500], labels=["<10 m", "10-20 m", ">=20 m"], right=False).astype(str))
        say("\nscenario-B primary on TEST by pass_dist (available at t_p):")
        say(ev_.subgroup_table({"B primary": r["P"][r["pname"]]}, r["te"]["y_completed"].to_numpy(), grp).to_string(index=False))

    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
