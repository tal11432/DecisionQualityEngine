"""Unit tests for the Decision Score (research/dq/dscore/score.py): the scale, the anchors, the gap and the perturbation scales."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "dq"))
from dscore import score as S
from dscore.core import Situation


def test_linear_scale_anchors():
    m = S.v13()
    assert S.q_to_score(0.0) == 50
    assert S.q_to_score(m["Q_hi"]) == 100 and S.q_to_score(5.0) == 100
    assert S.q_to_score(-m["Q_lo"]) == 0 and S.q_to_score(-5.0) == 0
    assert S.q_to_score(0.1) < S.q_to_score(0.3)


def test_rank_score_is_monotone_and_pinned():
    m = S.v13(); qs = np.linspace(-0.3, 1.0, 400); sc = [S.rank_score(q) for q in qs]
    assert all(b >= a for a, b in zip(sc, sc[1:]))
    assert S.rank_score(m["Q_hi"]) == 100 and S.rank_score(1.0) == 100                      # a 0.7+ xG situation is the top of the scale
    assert S.rank_score(-m["Q_lo"]) == 0                                                    # the one-on-one gift is the bottom
    assert 40 <= S.rank_score(float(np.median(np.sort(S._X["ref"])))) <= 60               # the median reference decision is near 50


def _sit(shooter, mates=(), opps=None):
    opps = np.array([[118.0, 40.0], [100.0, 20.0]]) if opps is None else np.array(opps)
    return Situation(shooter, opps, [np.array(m) for m in mates], [118.0, 40.0])


def test_best_shot_has_zero_gap():
    # a close, central shot with no visible teammates: the shot is the best option
    j = S.judge(_sit([112.0, 40.0]), ("shot",))
    assert j["gap_points"] == 0 and j["gap_xg"] == 0 and j["best"]["kind"] == "shot"
    assert j["score_chosen"] >= 90                                                         # a very good chance scores near the top


def test_gap_is_never_negative_and_worst_option_exists():
    rng = np.random.default_rng(0)
    for _ in range(15):
        a = [rng.uniform(85, 115), rng.uniform(20, 60)]; mates = [[rng.uniform(80, 118), rng.uniform(15, 65)] for _ in range(3)]
        j = S.judge(_sit(a, mates), ("shot",))
        assert j["gap_points"] >= 0 and j["gap_xg"] >= -1e-12 and j["score_worst"] <= j["score_chosen"] + 100


def test_a_poor_shot_is_below_a_better_pass():
    # a long shot from a wide angle while a teammate is free in front of goal
    j = S.judge(_sit([90.0, 70.0], mates=[[112.0, 42.0]], opps=[[118.0, 40.0], [114.0, 62.0]]), ("shot",))   # the free teammate is onside (second-last defender at x = 114)
    assert j["best"]["kind"] == "pass" and j["gap_xg"] > 0.03 and j["score_chosen"] < j["score_best"]


def test_perturbation_scales_multiply_values():
    base = S.judge(_sit([108.0, 40.0]), ("shot",)); half = S.judge(_sit([108.0, 40.0]), ("shot",), scales={"shot": 0.5})
    assert abs(half["q_chosen"] - 0.5 * base["q_chosen"]) < 1e-9 and half["score_chosen"] <= base["score_chosen"]


def test_chosen_pass_is_valued_at_its_real_target():
    s = _sit([100.0, 40.0], mates=[[108.0, 38.0]])
    good = S.judge(s, ("pass", [108.0, 38.0])); nowhere = S.judge(s, ("pass", [104.0, 75.0]))
    assert good["q_chosen"] > nowhere["q_chosen"] and good["score_chosen"] >= nowhere["score_chosen"]
