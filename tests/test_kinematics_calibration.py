import numpy as np
import pytest

from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.evaluation import auc
from src.ingestion.metrica import read_tracking
from src.processing.kinematics import decision_velocities, velocity_table
from src.schema import Pitch


def _td(tmp_path, xs, frames=None):
    """One Home player moving along x at given normalized positions; others static. Uses conftest writer."""
    from tests.conftest import _team_csv
    n = len(xs)
    frames = frames or list(range(1, n + 1))
    hp = [(str(i), f"Player{i}") for i in range(1, 6)]
    ap = [(str(i), f"Player{i}") for i in range(11, 16)]
    left = [(0.3, 0.5)] * 5
    right = [(0.7, 0.5)] * 5
    hrows, arows = [], []
    for x, fr in zip(xs, frames):
        h = [(x, 0.5)] + left[1:]
        hrows.append((1, fr, round(fr * 0.04, 2), h, (0.5, 0.5)))
        arows.append((1, fr, round(fr * 0.04, 2), right, (0.5, 0.5)))
    _team_csv(tmp_path / "h.csv", "Home", hp, hrows)
    _team_csv(tmp_path / "a.csv", "Away", ap, arows)
    return read_tracking(tmp_path / "h.csv", tmp_path / "a.csv", Pitch())


def test_velocity_central_difference(tmp_path):
    # 0.01 normalized per frame = 1.05 m per 0.04 s = 26.25 m/s (synthetic, just to check arithmetic)
    td = _td(tmp_path, [0.30, 0.31, 0.32, 0.33, 0.34])
    keys, vx, vy = velocity_table(td, 1)
    j = keys.index(("Home", "Player1"))
    assert np.isnan(vx[0, j]) and np.isnan(vx[-1, j])
    assert vx[2, j] == pytest.approx(1.05 / 0.04) and vy[2, j] == pytest.approx(0)
    assert vx[2, keys.index(("Home", "Player2"))] == pytest.approx(0)
    v = decision_velocities(td, 2, 1).set_index("player_id")   # causal: (pos[2]-pos[1])/dt
    assert v.loc["Player1", "vx"] == pytest.approx(26.25)


def test_velocity_nan_across_frame_gap(tmp_path):
    td = _td(tmp_path, [0.30, 0.31, 0.32, 0.33, 0.34], frames=[1, 2, 4, 5, 6])
    _, vx, _ = velocity_table(td, 1)
    assert np.isnan(vx[1, 0]) and np.isnan(vx[2, 0])  # windows spanning the gap
    assert decision_velocities(td, 1, 1)["vx"].notna().all()     # rows 0->1: frames 1,2, no gap
    assert np.isnan(decision_velocities(td, 2, 1)["vx"]).all()   # frames 2->4 span the gap
    with pytest.raises(ValueError):
        velocity_table(td, 0)


def test_velocity_aware_margin():
    P = FeasibilityParams(10.0, 5.0, 1.0, 101)
    opp = np.array([[10.0, 10.0]])
    static = interception_margin((0, 0), (20, 0), opp, P)["margin_s"]
    toward = interception_margin((0, 0), (20, 0), opp, P, np.array([[0.0, -5.0]]))["margin_s"]
    away = interception_margin((0, 0), (20, 0), opp, P, np.array([[0.0, 5.0]]))["margin_s"]
    nanv = interception_margin((0, 0), (20, 0), opp, P, np.array([[np.nan, np.nan]]))["margin_s"]
    assert toward > static > away and nanv == static


def test_auc():
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.3, 0.4]), y) == 1.0
    assert auc(np.array([0.4, 0.3, 0.2, 0.1]), y) == 0.0
    assert auc(np.array([1, 1, 1, 1.0]), y) == 0.5
    assert np.isnan(auc(np.array([1.0, 2.0]), np.array([1, 1])))
