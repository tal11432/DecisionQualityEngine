"""Metrics, calibration assessment, bootstrap comparisons and selection-bias diagnostics.

Conventions: ``p`` = predicted P(completed); y = 1 completed. The MINORITY class (intercepted) is the
event of interest for PR-AUC and for thresholded precision/recall/F1 (score = 1 - p). ROC-AUC is symmetric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import MIN_CLASS_FOR_CI, auc, boot_indices, confusion

EPS = 1e-12


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    """AUC for ranking completed above intercepted (== AUC for ranking intercepted by 1-p)."""
    return auc(np.asarray(p, float), np.asarray(y))


def pr_auc_intercepted(p: np.ndarray, y: np.ndarray) -> float:
    """Average precision for the minority class (intercepted = 1 - y) scored by 1 - p. Ties are grouped."""
    s, pos = 1 - np.asarray(p, float), (1 - np.asarray(y)).astype(bool)
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    s, pos = s[order], pos[order]
    tp = np.cumsum(pos)
    k = np.arange(1, len(s) + 1)
    last = np.r_[s[1:] != s[:-1], True]           # last index of each tie group
    prec, rec = tp[last] / k[last], tp[last] / pos.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


METRICS = {"roc_auc": roc_auc, "pr_auc_intercepted": pr_auc_intercepted, "brier": brier, "log_loss": log_loss}


def metric_ci(name: str, p, y, idx: np.ndarray) -> tuple[float, float, float]:
    f = METRICS[name]
    p, y = np.asarray(p, float), np.asarray(y)
    b = np.array([f(p[i], y[i]) for i in idx])
    b = b[np.isfinite(b)]
    return f(p, y), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_metric_diff(name: str, p_a, p_b, y, idx: np.ndarray) -> dict:
    """metric(a) - metric(b) on identical samples (paired bootstrap). For brier/log_loss lower is better."""
    f = METRICS[name]
    a, b, y = np.asarray(p_a, float), np.asarray(p_b, float), np.asarray(y)
    d = np.array([f(a[i], y[i]) - f(b[i], y[i]) for i in idx])
    d = d[np.isfinite(d)]
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"diff": f(a, y) - f(b, y), "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_assessment(p: np.ndarray, y: np.ndarray, fit_logistic) -> dict:
    """Equal-count reliability table on P(intercepted) plus calibration intercept/slope (diagnostic fit on the
    evaluation set; NEVER applied back). Bins are limited so that each holds ~>=10 intercepted passes."""
    p_int, y_int = 1 - np.asarray(p, float), 1 - np.asarray(y)
    n_pos = int(y_int.sum())
    nb = int(min(5, n_pos // 10))
    out = {"n_bins": nb, "table": pd.DataFrame(), "slope": float("nan"), "intercept": float("nan")}
    if nb >= 2:
        order = np.argsort(p_int, kind="stable")
        rows = []
        for chunk in np.array_split(order, nb):
            rows.append((len(chunk), float(p_int[chunk].mean()), float(y_int[chunk].mean())))
        out["table"] = pd.DataFrame(rows, columns=["n", "mean_predicted_P(intercepted)", "observed_rate"])
    if n_pos >= MIN_CLASS_FOR_CI and len(y_int) - n_pos >= MIN_CLASS_FOR_CI:
        z = logit(p_int)[:, None]
        w, b = fit_logistic(z, y_int.astype(float), 1e-6)
        out["slope"], out["intercept"] = float(w[0]), float(b)
    return out


def reliability_plot(path: str, series: dict[str, tuple[np.ndarray, np.ndarray]], fit_logistic, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    top = 0.05
    for name, (p, y) in series.items():
        t = calibration_assessment(p, y, fit_logistic)["table"]
        if len(t):
            top = max(top, float(t.iloc[:, 1].max()), float(t["observed_rate"].max()))
            ax.plot(t.iloc[:, 1], t["observed_rate"], "o-", label=f"{name} (bins n={t['n'].min()}-{t['n'].max()})")
    lim = 1.15 * top
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_xlabel("mean predicted P(intercepted)")
    ax.set_ylabel("observed interception rate")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def threshold_f1(p: np.ndarray, y: np.ndarray) -> float:
    """tau maximising F1 for 'intercepted' (flag = p_completed <= tau) on the given (training/OOF) data."""
    s, yi = 1 - np.asarray(p, float), 1 - np.asarray(y)
    cands = np.unique(s)
    best = max(cands, key=lambda t: confusion(s, yi, t)["f1"])
    return float(1 - best)


def confusion_at(p: np.ndarray, y: np.ndarray, tau: float) -> dict:
    return confusion(1 - np.asarray(p, float), 1 - np.asarray(y), 1 - tau)


def subgroup_table(p_by_model: dict[str, np.ndarray], y: np.ndarray, groups: pd.Series, n_boot: int = 1000) -> pd.DataFrame:
    """ROC-AUC per subgroup with CI; 'n/a' if either class has < 15 samples."""
    y = np.asarray(y)
    rows = []
    for g in sorted(groups.dropna().unique(), key=str):
        m = (groups == g).to_numpy()
        n_int, n_comp = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        row = {"group": str(g), "n_intercepted": n_int, "n_completed": n_comp}
        idx = boot_indices(int(m.sum()), n_boot, 1) if min(n_int, n_comp) >= MIN_CLASS_FOR_CI else None
        for name, p in p_by_model.items():
            if idx is None:
                row[name] = "n/a"
            else:
                a, lo, hi = metric_ci("roc_auc", np.asarray(p)[m], y[m], idx)
                row[name] = f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- selection bias
def standardized_difference(a: pd.Series, b: pd.Series) -> float:
    """(mean_a - mean_b) / pooled SD."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else float("nan")


def retained_vs_excluded(table_a: pd.DataFrame, retained_ids: set, features: list[str]) -> pd.DataFrame:
    """Decision-time (scenario A) features of passes retained vs excluded by the scenario-B target filter,
    per outcome class: mean retained, mean excluded, standardized difference."""
    t = table_a.assign(retained=table_a["event_idx"].isin(retained_ids))
    rows = []
    for cls, lab in ((1, "completed"), (0, "intercepted")):
        s = t[t["y_completed"] == cls]
        for f in features:
            r, e = s.loc[s.retained, f], s.loc[~s.retained, f]
            rows.append((lab, f, int(len(r)), int(len(e)), float(r.mean()), float(e.mean()), standardized_difference(r, e)))
    return pd.DataFrame(rows, columns=["class", "feature", "n_retained", "n_excluded", "mean_retained", "mean_excluded", "std_diff"])
