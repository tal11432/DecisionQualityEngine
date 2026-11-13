"""SkillCorner release-audit detection rule on synthetic 10 Hz tracking (no real data)."""
import numpy as np
import pandas as pd
import pytest

import src.skillcorner_release_audit as sa


def trk(n=80, release=40, pass_speed_per_frame=1.0, detected=True, bounce=False):
    """Passer 7 stands at the origin; the ball is at its feet until ``release``, then moves away at pass_speed_per_frame m/frame."""
    ball = np.zeros((n, 2))
    for i in range(n):
        ball[i, 0] = 0.3 + max(0, i - release) * pass_speed_per_frame
    if bounce:                                                      # second departure: ball returns to the passer and leaves again
        ball[52:, 0] = 0.3 + np.maximum(0, np.arange(52, n) - 60) * 1.0
    return {"frame0": 0, "n": n, "first_play": 0, "ball": ball, "ball_det": np.full(n, detected),
            "players": {7: [np.zeros((n, 2)), np.full(n, detected)]}}


def ev(fs, fe):
    return pd.Series({"frame_start": fs, "frame_end": fe, "player_id": 7}, name=0)


def test_departure_rule_needs_lookahead_frames_outside_contact():
    d = np.array([0.2, 0.3, 0.4, 3.0, 4.0, 5.0, 6.0])
    assert sa.departures(d) == [3]
    assert sa.departures(np.array([0.2, 0.3, 3.0, 0.5, 0.4, 0.4, 0.4])) == []         # a one-frame excursion is not a departure
    assert sa.departures(np.array([0.2, 0.3, np.nan, 3.0, 4.0, 5.0])) == []           # missing data is never interpreted


def test_release_is_the_last_contact_frame_and_error_is_measured_against_event_frames():
    r = sa.analyse_pass(trk(release=40), ev(30, 40))
    # the ball starts moving after frame 40 but is still 1.3 m away at frame 41 (inside the 1.5 m contact radius): release = last frame in contact = 41
    assert r["identified"] and r["release_frame"] == 41
    assert r["err_end"] == r["release_frame"] - 40 and r["err_start"] == r["release_frame"] - 30
    assert abs(r["err_end"]) <= 1 and r["err_start"] >= 8                              # release matches frame_end, not frame_start
    assert r["speed_03s"] > 5 and r["deflect_deg"] < 60


def test_extrapolated_positions_are_not_independent_evidence():
    r = sa.analyse_pass(trk(detected=False), ev(30, 40))
    assert not r.get("identified") and "extrapolated" in r["reason"]


def test_several_departures_are_ambiguous_and_never_resolved_with_event_frames():
    r = sa.analyse_pass(trk(bounce=True), ev(40, 50))
    assert not r.get("identified") and r["reason"].startswith("ambiguous")


def test_no_ball_contact_means_no_identifiable_release():
    t = trk()
    t["ball"] = t["ball"] + 10.0                                                       # ball never near the passer
    assert "no departure" in sa.analyse_pass(t, ev(30, 40))["reason"]


def test_wilson_interval_brackets_the_estimate():
    lo, hi = sa.wilson(62, 95)
    assert lo < 62 / 95 < hi and 0.5 < lo and hi < 0.8
    assert all(np.isnan(sa.wilson(0, 0)))
