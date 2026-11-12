import numpy as np
import pandas as pd
import pytest

from src.evaluation import (auc_ci, best_f1_threshold, boot_indices, clean, confusion, paired_auc_diff,
                            subgroup_auc)


def test_confusion_and_f1():
    s, y = np.array([0.9, 0.8, 0.2, 0.1, 0.6]), np.array([1, 0, 1, 0, 0])
    c = confusion(s, y, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 2, 1, 1)
    assert c["precision"] == pytest.approx(1 / 3) and c["recall"] == 0.5 and c["f1"] == pytest.approx(0.4)
    assert confusion(s, y, 5.0)["f1"] == 0.0


def test_best_threshold_is_dev_derived():
    s, y = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])
    assert best_f1_threshold(s, y) == 3.0


def test_bootstrap_is_seeded_and_ci_brackets_point():
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.3).astype(int)
    s = y * 0.8 + rng.normal(0, 1, 200)
    a, lo, hi = auc_ci(s, y, boot_indices(200, 300, seed=3))
    a2, lo2, hi2 = auc_ci(s, y, boot_indices(200, 300, seed=3))
    assert (a, lo, hi) == (a2, lo2, hi2) and lo < a < hi


def test_paired_diff_detects_real_gap_and_null():
    rng = np.random.default_rng(2)
    y = (rng.random(400) < 0.3).astype(int)
    good, noise = y * 2.0 + rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    idx = boot_indices(400, 300)
    assert paired_auc_diff(good, noise, y, idx)["significant"]
    assert not paired_auc_diff(good, good + 1e-9, y, idx)["significant"]


def test_subgroup_requires_minimum_class_counts():
    y = np.array([1] * 20 + [0] * 40 + [1] * 3 + [0] * 40)
    g = pd.Series(["big"] * 60 + ["small"] * 43)
    t = subgroup_auc({"m": np.arange(len(y), dtype=float)}, y, g, n_boot=50).set_index("group")
    assert t.loc["small", "m"] == "n/a" and t.loc["big", "m"] != "n/a"


def test_clean_maps_no_opponent_margin_to_fixed_sentinel():
    assert clean(np.array([-np.inf, 1.0]))[0] == -1e3
