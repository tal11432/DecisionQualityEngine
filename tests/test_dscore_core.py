"""Unit tests for the shared Decision Score helpers (research/dq/dscore/core.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "dq"))
from dscore import core


def test_triangle_membership():
    shooter = np.array([108.0, 40.0])
    assert core.tri(np.array([114.0, 40.0]), shooter)          # between shooter and goal
    assert not core.tri(np.array([100.0, 40.0]), shooter)      # behind the shooter
    assert not core.tri(np.array([112.0, 60.0]), shooter)      # far wide


def test_offside_status_rules():
    assert core.offside_status(5.0, 100.0, 5) == "certain onside"
    assert core.offside_status(0.5, 100.0, 11).startswith("DOUBT")
    assert core.offside_status(-0.5, 100.0, 11).startswith("DOUBT")
    assert core.offside_status(-3.0, 100.0, 8).startswith("DOUBT")        # a hidden defender can only help the attacker
    assert core.offside_status(-3.0, 100.0, 11) == "certain offside"
    assert core.offside_status(-3.0, 55.0, 11) == "certain onside"        # own half


def test_verdict_needs_ratio_and_absolute_floor():
    assert core.verdict(0.06, 0.03, 5) == "no clear difference"           # 0.03 vs 0.06: three hundredths of a goal is noise
    assert core.verdict(0.50, 0.10, 5) == "pass clearly better"
    assert core.verdict(0.05, 0.40, 5) == "shot clearly better"
    assert core.verdict(0.05, 0.085, 5) == "leans shot"                    # margin 0.085 - 0.065 = 0.02: above the lean level (0.015), below the floor (0.03)
    assert core.verdict(0.05, 0.075, 5) == "no clear difference"           # margin 0.01: below the lean level
    # long passes use the stricter floor
    assert core.verdict(0.30, 0.24, 20) in ("leans pass", "no clear difference")


def test_hazard_falls_with_a_defender_on_the_line():
    a, end = np.array([90.0, 40.0]), np.array([105.0, 40.0])
    clear = core.xp_hazard(a, end, np.array([[95.0, 70.0]]))
    blocked = core.xp_hazard(a, end, np.array([[98.0, 40.5]]))
    assert 0 <= blocked < clear <= 1
    assert core.xp_hazard(a, a, np.array([[95.0, 40.0]])) == 1.0           # zero-length pass


def test_shot_value_falls_with_distance():
    O = np.array([[118.0, 40.0], [100.0, 30.0]]); gk = np.array([118.0, 40.0])
    near = core.shot_value(core.Situation([110.0, 40.0], O, [], gk))[2]
    far = core.shot_value(core.Situation([90.0, 40.0], O, [], gk))[2]
    assert near > far > 0


def test_pass_options_respect_range_and_offside():
    a = np.array([100.0, 40.0]); gk = np.array([118.0, 40.0]); O = np.array([[116.0, 40.0], [112.0, 45.0], [118.0, 40.0]])
    sit = core.Situation(a, O, [[101.0, 41.0], [108.0, 38.0], [119.0, 36.0], [50.0, 40.0]], gk)     # too close, onside, beyond the last defender, too far (50 units = 43.75 m)
    opts = core.pass_options(sit)
    assert len(opts) == 2
    statuses = {round(o["g"]): o["status"] for o in opts}
    assert any(s == "certain onside" for s in statuses.values())
    live, sure, gone = core.rank_options(opts)
    assert len(live) + len(gone) == 2 and all(o in live for o in sure)
