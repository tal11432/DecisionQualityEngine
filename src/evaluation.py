"""Metrics for a binary score (here: geometric interception margin; NOT a probability).

Positive class = intercepted (1). Higher score = more interceptable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CLASS_FOR_CI = 15      # a subgroup needs at least this many of EACH class to report AUC
NO_OPPONENT_SENTINEL = -1e3  # margin -inf (no opponent) mapped to a fixed finite score


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score_pos > score_neg), ties = 0.5 (Mann-Whitney). NaN if a class is empty."""
    score, label = np.asarray(score, float), np.asarray(label)
    n1, n0 = int((label == 1).sum()), int((label == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def clean(score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, float)
    return np.where(np.isfinite(s), s, NO_OPPONENT_SENTINEL)


def boot_indices(n: int, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def auc_ci(score, label, idx: np.ndarray) -> tuple[float, float, float]:
    """(AUC, lo, hi) with a percentile bootstrap (95%) over decisions, using shared resample indices."""
    score, label = np.asarray(score, float), np.asarray(label)
    b = np.array([auc(score[i], label[i]) for i in idx])
    b = b[np.isfinite(b)]
    return auc(score, label), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_auc_diff(score_a, score_b, label, idx: np.ndarray) -> dict:
    """AUC(a) - AUC(b) on the SAME samples, paired bootstrap. ``significant`` = 95% CI excludes 0."""
    a, b, y = np.asarray(score_a, float), np.asarray(score_b, float), np.asarray(label)
    d = np.array([auc(a[i], y[i]) - auc(b[i], y[i]) for i in idx])
    d = d[np.isfinite(d)]
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"diff": auc(a, y) - auc(b, y), "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}


def confusion(score, label, threshold: float) -> dict:
    """Flag = score >= threshold. Positive = intercepted."""
    s, y = np.asarray(score, float), np.asarray(label)
    pred = s >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def best_f1_threshold(score, label) -> float:
    """Threshold maximising F1 on the given (development) samples."""
    s = np.asarray(score, float)
    cands = np.unique(s)
    return float(max(cands, key=lambda t: confusion(s, label, t)["f1"]))


def subgroup_auc(scores: dict[str, np.ndarray], label: np.ndarray, groups: pd.Series, n_boot: int = 1000,
                 seed: int = 1) -> pd.DataFrame:
    """AUC (+CI) per subgroup and model; 'n/a' when either class has < MIN_CLASS_FOR_CI samples."""
    y = np.asarray(label)
    rows = []
    for g in groups.dropna().unique():
        m = (groups == g).to_numpy()
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        row = {"group": str(g), "n_intercepted": n1, "n_completed": n0}
        idx = boot_indices(int(m.sum()), n_boot, seed) if min(n1, n0) >= MIN_CLASS_FOR_CI else None
        for name, s in scores.items():
            if idx is None:
                row[name] = "n/a"
            else:
                a, lo, hi = auc_ci(np.asarray(s)[m], y[m], idx)
                row[name] = f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)
