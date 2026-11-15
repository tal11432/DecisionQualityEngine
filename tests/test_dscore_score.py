"""Unit tests for the Decision Score (research/dq/dscore/score.py), amendment E of PREREG_DQ_v13.txt: the ABSOLUTE scale of the expected value, the user's anchor cases, the gap and the face-validity checks."""
import inspect, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "dq"))
from dscore import score as S
from dscore.core import Situation


def test_scale_anchors():
    m = S.v13(); c = S.v15()["C"]
    assert S.q_to_score(m["Q_hi"]) == 100 and S.q_to_score(5.0) == 100                      # 0.7 xG and above is the top of the scale
    assert S.q_to_score(-c) == 0 and S.q_to_score(-5.0) == 0                                # the bottom: a counter-attack handed to the opponent
    assert S.q_to_score(0.1) < S.q_to_score(0.3) < S.q_to_score(0.6)                        # monotone in the expected value


def test_the_users_cases_on_the_scale():
    # a 0.01 xG shot is never high (even if it goes in), a 0.99 xG chance is high (even if it is missed): the score only sees the expected value
    assert S.q_to_score(0.01) < 30 and S.q_to_score(0.99) >= 95
    assert S.q_to_score(0.32) > S.q_to_score(0.08) > S.q_to_score(0.01)
    assert S.q_to_score(0.32) - S.q_to_score(0.06) >= 20                                    # 0.32 xG is far above 0.06 xG (the percentile scale gave 99 / 88)


def test_the_score_never_sees_the_outcome():
    params = set(inspect.signature(S.judge).parameters)
    assert not ({"goal", "outcome", "scored", "result"} & params)                            # no outcome flag can enter
    s = _sit([108.0, 40.0]); assert S.judge(s, ("shot",))["score_chosen"] == S.judge(s, ("shot",))["score_chosen"] == S.q_to_score(S.judge(s, ("shot",))["q_chosen"])


def _sit(shooter, mates=(), opps=None):
    opps = np.array([[118.0, 40.0], [100.0, 20.0]]) if opps is None else np.array(opps)
    return Situation(shooter, opps, [np.array(m) for m in mates], [118.0, 40.0])


def test_best_shot_has_zero_gap():
    # a close, central shot with no visible teammates: the shot is the best option
    j = S.judge(_sit([112.0, 40.0]), ("shot",))
    assert j["gap_points"] == 0 and j["gap_xg"] == 0 and j["best"]["kind"] == "shot"
    assert j["score_chosen"] == S.q_to_score(j["q_chosen"]) >= 40                           # a good chance is clearly above the neutral level


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


# ---------------------------------------------------------------- face validity (amendment E, gates (c)-(e))
def test_a_covered_lane_scores_below_a_free_lane_and_above_nowhere():
    mates = [[110.0, 36.0]]; free = Situation([98.0, 40.0], np.array([[118.0, 40.0], [92.0, 20.0]]), [np.array(m) for m in mates], [118.0, 40.0])
    covered = Situation([98.0, 40.0], np.array([[118.0, 40.0], [104.0, 38.0], [105.0, 37.0]]), [np.array(m) for m in mates], [118.0, 40.0])
    f = S.pass_value(free, [110.0, 36.0]); c = S.pass_value(covered, [110.0, 36.0]); nw = S.pass_value(covered, [100.0, 75.0])
    assert f["xp"] > c["xp"]                                                                # the covered lane is less likely to arrive
    assert S.q_to_score(c["v"]) < S.q_to_score(f["v"])                                      # lower success probability lowers the score even if the receiver is in a scoring position
    assert S.q_to_score(c["v"]) >= S.q_to_score(nw["v"])                                    # but arrival creates a chance, so it is not scored like a pass to nowhere


@pytest.mark.xfail(strict=False, reason="amendment F: one 360 frame (visible players, no velocities) does not identify a counter-attack with a numerical advantage (G-counter failed); a model that can see it will turn this into an XPASS")
def test_a_loss_with_a_counter_scores_low_and_more_than_without():
    # edge of our own box, three opponents beyond the ball against one of ours covering (a counter-attack with a numerical advantage) vs nobody beyond the ball
    O_c = np.array([[10.0, 36.0], [12.0, 44.0], [14.0, 40.0], [40.0, 40.0]]); T_c = np.array([[4.0, 40.0], [60.0, 30.0]])
    O_n = np.array([[40.0, 40.0], [45.0, 30.0]]); T_n = np.array([[4.0, 40.0], [10.0, 30.0], [12.0, 50.0]])
    l_c = S.loss_danger(np.array([18.0, 40.0]), O_c, T_c, "pass"); l_n = S.loss_danger(np.array([18.0, 40.0]), O_n, T_n, "pass")
    assert l_c > l_n + 0.02                                                                 # the counter-attack is clearly more dangerous
    assert S.q_to_score(-l_c) <= 15 and S.q_to_score(-l_c) < S.q_to_score(-l_n)             # pre-registered: <= 15


@pytest.mark.xfail(strict=False, reason="amendment F: same limitation (the modelled cost of a loss does not rise with the opponents' visible numerical advantage)")
def test_a_crowded_carry_is_worse_when_a_loss_means_a_counter():
    crowd = [[101.0, 40.0], [102.0, 38.0], [100.0, 43.0], [99.0, 36.0], [118.0, 40.0]]
    counter = Situation([95.0, 40.0], np.array(crowd + [[70.0, 40.0], [72.0, 30.0], [75.0, 50.0]]), [np.array([97.0, 55.0])], [118.0, 40.0])      # three opponents beyond the ball, nobody of ours behind it
    safe = Situation([95.0, 40.0], np.array(crowd), [np.array(m) for m in ([97.0, 55.0], [70.0, 40.0], [72.0, 30.0], [75.0, 50.0])], [118.0, 40.0])
    c = S.carry_value(counter, [100.0, 40.0]); s_ = S.carry_value(safe, [100.0, 40.0])
    assert c["loss"] > s_["loss"] and c["v"] < s_["v"] and S.q_to_score(c["v"]) < S.q_to_score(s_["v"])
