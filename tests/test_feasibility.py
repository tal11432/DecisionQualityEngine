import numpy as np
import pandas as pd
import pytest

from src.analytics.feasibility import (FeasibilityParams, interception_margin, pass_options,
                                       receiver_arrival_margin)
from src.schema import Snapshot

P = FeasibilityParams(ball_speed=10.0, player_speed=5.0, reaction_time=0.0, lane_samples=101)


def test_defender_on_the_lane_intercepts():
    r = interception_margin((0, 0), (20, 0), np.array([[10.0, 0.0]]), P)
    assert r["interceptable"] and r["margin_s"] == pytest.approx(1.0, abs=0.05)  # ball needs 1s, defender 0s at x=10


def test_far_defender_cannot_intercept():
    r = interception_margin((0, 0), (20, 0), np.array([[10.0, 40.0]]), P)
    assert not r["interceptable"] and r["margin_s"] < 0


def test_known_boundary():
    # Defender 5 m off the lane midpoint: reaches x=10 in 1 s, ball in 1 s -> margin ~0 at that point,
    # but the best lane point for the defender is earlier in the lane.
    near = interception_margin((0, 0), (20, 0), np.array([[10.0, 5.0]]), P)
    far = interception_margin((0, 0), (20, 0), np.array([[10.0, 12.0]]), P)
    assert near["margin_s"] > far["margin_s"] and near["interceptable"] and not far["interceptable"]


def test_faster_ball_and_reaction_time_help_passer():
    d = np.array([[10.0, 4.0]])
    slow = interception_margin((0, 0), (20, 0), d, FeasibilityParams(8, 5, 0.0, 101))["margin_s"]
    fast = interception_margin((0, 0), (20, 0), d, FeasibilityParams(25, 5, 0.0, 101))["margin_s"]
    lazy = interception_margin((0, 0), (20, 0), d, FeasibilityParams(8, 5, 1.0, 101))["margin_s"]
    assert fast < slow and lazy < slow


def test_no_or_nan_opponents():
    for opp in (np.empty((0, 2)), np.array([[np.nan, np.nan]])):
        r = interception_margin((0, 0), (10, 0), opp, P)
        assert not r["interceptable"] and r["margin_s"] == -np.inf and r["threat_idx"] is None


def test_zero_length_pass_does_not_crash():
    r = interception_margin((1, 1), (1, 1), np.array([[1.0, 1.0]]), P)
    assert r["pass_length"] == 0


def test_receiver_arrival():
    assert receiver_arrival_margin((0, 0), (20, 0), (20, 0), P) == pytest.approx(2.0)
    assert receiver_arrival_margin((0, 0), (20, 0), (20, 30), P) < 0


def test_invalid_params():
    with pytest.raises(ValueError):
        FeasibilityParams(ball_speed=0)


def test_pass_options_table():
    p = pd.DataFrame([("Home", "H1", "1", 0.0, 0.0), ("Home", "H2", "2", 20.0, 0.0), ("Home", "H3", "3", 0.0, 20.0),
                      ("Away", "A1", "1", 10.0, 0.5)], columns=["team", "player_id", "jersey", "x", "y"])
    t = pass_options(Snapshot(p, (0, 0), 1, 1, 0.04), "H1", P).set_index("receiver")
    assert t.loc["H2", "interceptable"] and t.loc["H2", "threat_player"] == "A1"
    assert not t.loc["H3", "interceptable"]
    with pytest.raises(KeyError):
        pass_options(Snapshot(p, (0, 0), 1, 1, 0.04), "nobody")
