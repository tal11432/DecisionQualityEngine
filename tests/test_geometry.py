import math

import numpy as np
import pandas as pd
import pytest

from src.analytics import geometry as g
from src.analytics.interfaces import decision_quality
from src.schema import Snapshot


def snap():
    p = pd.DataFrame([
        ("Home", "H1", "1", 0.0, 0.0), ("Home", "H2", "2", 3.0, 4.0), ("Home", "H3", "3", 30.0, 0.0),
        ("Away", "A1", "1", 1.0, 0.0), ("Away", "A2", "2", 10.0, 10.0)],
        columns=["team", "player_id", "jersey", "x", "y"])
    return Snapshot(p, (0.0, 3.0), 1, 1, 0.04)


def test_pairwise_distances():
    d = g.pairwise_distances(np.array([[0, 0], [3, 4], [0, 0]]))
    assert d[0, 1] == 5 and d[0, 2] == 0 and np.allclose(d, d.T)


def test_distance_to_point_and_goal():
    assert g.distances_to_point(np.array([[0, 0], [3, 4]]), (0, 0)).tolist() == [0, 5]
    assert g.distance_to_goal(np.array([[42.5, 0.0]]))[0] == pytest.approx(10)


def test_angle_to_goal():
    ang = g.angle_to_goal(np.array([[52.5 - 11, 0.0], [52.5, 0.0], [30.0, 0.0], [52.5, 30.0]]))
    assert ang[0] == pytest.approx(2 * math.atan(3.66 / 11))  # penalty-spot aperture
    assert ang[1] == pytest.approx(math.pi)                  # on the line between the posts
    assert ang[2] < ang[0]                                   # further away -> narrower
    assert ang[3] == pytest.approx(0.0, abs=1e-9)            # on the goal line, collinear with the goal line, outside posts


def test_nearest_defender_and_teammate():
    s = snap()
    assert g.nearest_defender(s, "H1") == ("A1", 1.0)
    pid, d = g.nearest_teammate(s, "H1")
    assert (pid, d) == ("H2", 5.0)
    with pytest.raises(KeyError):
        g.nearest_defender(s, "nobody")


def test_nearest_with_no_candidates():
    s = snap()
    s.players = s.players[s.players.team == "Home"]
    assert g.nearest_defender(s, "H1")[0] is None


def test_passing_lane_obstruction():
    defs = np.array([[5.0, 0.5], [5.0, 5.0], [-5.0, 0.0]])
    r = g.passing_lane_obstruction((0, 0), (10, 0), defs, lane_width=2.0)
    assert r["obstructed"] and r["n_blockers"] == 1 and r["min_distance"] == pytest.approx(0.5)
    # behind the passer: distance measured to the segment END, not the infinite line
    r2 = g.passing_lane_obstruction((0, 0), (10, 0), np.array([[-5.0, 0.0]]), 2.0)
    assert not r2["obstructed"] and r2["min_distance"] == pytest.approx(5.0)
    assert g.passing_lane_obstruction((0, 0), (10, 0), np.empty((0, 2)))["n_blockers"] == 0
    assert not g.passing_lane_obstruction((0, 0), (10, 0), np.array([[np.nan, np.nan]]))["obstructed"]


def test_snapshot_features_columns_and_values():
    f = g.snapshot_features(snap())
    h1 = f[f.player_id == "H1"].iloc[0]
    assert h1["dist_ball"] == pytest.approx(3.0)
    assert h1["nearest_opp_dist"] == 1.0 and h1["nearest_mate_dist"] == 5.0


def test_decision_quality_arithmetic():
    assert decision_quality(0.2, [0.1, 0.5]) == pytest.approx(-0.3)
    with pytest.raises(ValueError):
        decision_quality(0.2, [])
