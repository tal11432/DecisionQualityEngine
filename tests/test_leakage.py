"""Regression tests: outcome leakage must not re-enter the pass-feasibility pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.calibration import Model, score_margin
from src.analytics.feasibility import FeasibilityParams
from src.pass_dataset import DecisionContract, build_decision_samples
from src.processing.kinematics import decision_velocities, velocity_table
from src.schema import Pitch, TrackingData

REL = 40  # release row == frame number
N = 80
TOL = 0.02


def make_td(heading=(1.0, 0.0), h4=(-20.0, 20.0), ball_after_nan=False) -> TrackingData:
    """Passer H1 at origin holds the ball until row 40, then the ball leaves along ``heading`` at 12 m/s.
    H2 (15, .5) is the natural target. Opponent A1 drifts at 2 m/s in -y near the lane."""
    t = np.arange(N) * 0.04
    rows = np.arange(N)
    hd = np.array(heading) / np.hypot(*heading)
    after = np.clip(rows - REL, 0, None) * 0.04 * 12.0
    f = {"period": np.ones(N, int), "frame": rows, "time_s": t,
         "ball_x": 0.2 + after * hd[0], "ball_y": 0.0 + after * hd[1]}
    if ball_after_nan:
        f["ball_x"] = np.where(rows > REL, np.nan, f["ball_x"])
    pos = {("Home", "H1"): ((0.0, 0.0),), ("Home", "H2"): ((15.0, 0.5),), ("Home", "H3"): ((0.0, 15.0),),
           ("Home", "H4"): (h4,), ("Away", "A1"): None, ("Away", "A2"): ((30.0, 30.0),), ("Away", "A3"): ((-30.0, -30.0),)}
    for (team, pid), v in pos.items():
        if v is None:   # moving opponent
            f[f"{team}__{pid}__x"] = np.full(N, 7.5)
            f[f"{team}__{pid}__y"] = 3.0 - 0.08 * rows
        else:
            f[f"{team}__{pid}__x"] = np.full(N, v[0][0])
            f[f"{team}__{pid}__y"] = np.full(N, v[0][1])
    players = pd.DataFrame([(t_, p, None, False) for (t_, p) in pos], columns=["team", "player_id", "jersey", "id_missing"])
    return TrackingData(pd.DataFrame(f), players, Pitch(), "synthetic")


def events(kind="PASS", end=(15.0, 0.5), to="H2", time=REL * 0.04) -> pd.DataFrame:
    subtype = None if kind == "PASS" else "INTERCEPTION"
    return pd.DataFrame([dict(team="Home", type=kind, subtype=subtype, period=1, start_frame=REL, start_time_s=time,
                              end_frame=60, end_time_s=2.4, from_player="H1", to_player=to,
                              start_x=0.0, start_y=0.0, end_x=end[0], end_y=end[1])])


def build(td=None, ev=None, **kw):
    return build_decision_samples(td or make_td(), events() if ev is None else ev, TOL, DecisionContract(**kw))


M = Model("m", FeasibilityParams(12.0, 6.0, 0.3, 60), True)


def same(a, b):
    assert (a.event_idx, a.label, a.decision_row, a.receiver_id) == (b.event_idx, b.label, b.decision_row, b.receiver_id)
    assert a.origin == b.origin and a.target == b.target
    np.testing.assert_array_equal(a.opp_xy, b.opp_xy)
    np.testing.assert_array_equal(a.opp_v, b.opp_v)


def test_baseline_sample_is_built_from_decision_time_state():
    r = build()
    assert len(r.samples) == 1
    s = r.samples[0]
    assert s.decision_row == REL and s.receiver_id == "H2" and s.label == 0
    assert s.target == (15.0, 0.5) and s.origin == (0.0, 0.0)       # positions at t_d, not reception
    a1 = [v for v in s.opp_v if np.isclose(v[0], 0.0)]
    assert any(np.isclose(v[1], -2.0) for v in a1)                  # causal velocity of A1 = -2 m/s


def test_end_point_and_outcome_columns_do_not_affect_samples():
    base = build().samples[0]
    scrambled = events()
    for c in ("end_x", "end_y", "start_x", "start_y"):
        scrambled[c] = 999.0
    scrambled["end_time_s"], scrambled["end_frame"], scrambled["start_frame"] = np.nan, 12345, 12345
    scrambled["to_player"] = "A2"        # even the true receiver is not an input
    same(base, build(ev=scrambled).samples[0])


def test_interception_location_does_not_change_prediction():
    far = events("BALL LOST", end=(40.0, -20.0), to=None)
    on_target = events("BALL LOST", end=(15.0, 0.5), to=None)          # interception 'on' the receiver
    on_opp = events("BALL LOST", end=(7.5, 2.0), to=None)              # ... or exactly on the lurking defender
    scores = []
    for ev in (far, on_target, on_opp):
        s = build(ev=ev).samples
        assert len(s) == 1 and s[0].label == 1
        scores.append(score_margin(s, M)[0])
    assert scores[0] == scores[1] == scores[2]


def test_actual_endpoint_cannot_define_target():
    """Heading goes into space; an event end point sitting on a teammate must not rescue the sample."""
    td = make_td(heading=(0.6, -0.8))                                  # ~53 deg off H2
    r = build(td=td, ev=events("BALL LOST", end=(15.0, 0.5), to="H2"))
    assert r.samples == []
    assert r.exclusions["reason"].iloc[0].startswith("no teammate within heading cone")


def test_future_frames_do_not_change_inputs():
    base = build().samples[0]
    td = make_td()
    rng = np.random.default_rng(0)
    cols = [c for c in td.frames.columns if c.endswith(("__x", "__y"))]
    for c in cols:                                   # every player, every frame after the decision
        td.frames.loc[REL + 1:, c] = rng.normal(0, 30, N - REL - 1)
    td.frames.loc[REL + 7:, ["ball_x", "ball_y"]] = 1234.0   # ball beyond the label-side heading window
    same(base, build(td=td).samples[0])


def test_causal_velocity_ignores_future_but_central_does_not():
    td = make_td()
    col = td.col("Away", "A1", "y")
    j = list(td.players.player_id).index("A1")
    before_c = decision_velocities(td, REL, 5)["vy"].iloc[j]
    keys, _, vy0 = velocity_table(td, 2)
    td.frames.loc[REL + 1:, col] += 50.0
    assert decision_velocities(td, REL, 5)["vy"].iloc[j] == before_c
    assert velocity_table(td, 2)[2][REL, j] != vy0[REL, j]         # central estimator would leak


def test_velocity_uses_only_history():
    td = make_td()
    col = td.col("Away", "A1", "y")
    j = list(td.players.player_id).index("A1")
    base = decision_velocities(td, REL, 5)["vy"].iloc[j]
    td.frames.loc[REL - 5, col] += 1.0                            # change history -> must change
    assert decision_velocities(td, REL, 5)["vy"].iloc[j] != base


def test_completed_and_intercepted_follow_identical_rules():
    a = build(ev=events("PASS", to="H2")).samples[0]
    b = build(ev=events("BALL LOST", end=(70.0, 70.0), to=None)).samples[0]   # different outcome info entirely
    assert (a.label, b.label) == (0, 1)
    same(a.__class__(**{**a.__dict__, "label": 1}), b)


def test_ambiguous_target_is_excluded_not_guessed():
    r = build(td=make_td(h4=(25.0, 0.9)))                          # H4 collinear behind H2
    assert r.samples == [] and r.exclusions["reason"].iloc[0].startswith("ambiguous receiver")


def test_pass_into_space_excluded_with_reason_and_label_recorded():
    r = build(td=make_td(heading=(0.0, 1.0)), ev=events("BALL LOST", to=None))   # ball heads at H3? (0,15): in cone
    assert len(r.samples) == 1 and r.samples[0].receiver_id == "H3"
    r = build(td=make_td(heading=(-1.0, 0.0)), ev=events("BALL LOST", to=None))  # nobody behind the passer
    assert r.samples == [] and r.exclusions["label"].iloc[0] == 1


def test_missing_post_release_ball_excluded():
    r = build(td=make_td(ball_after_nan=True))
    assert r.samples == [] and "post-release ball track" in r.exclusions["reason"].iloc[0]


def test_unsynchronizable_event_is_excluded():
    r = build(ev=events(time=999.0))
    assert r.samples == [] and r.exclusions["reason"].iloc[0].startswith("no tracking frame")


def test_possession_filter_is_decision_time():
    td = make_td()
    td.frames.loc[REL, "ball_x"] = 10.0                            # ball far from passer at t_d
    r = build(td=td)
    assert r.samples == [] and r.exclusions["reason"].iloc[0] == "ball not with passer at decision frame"


def test_decision_offset_moves_inputs_earlier_not_later():
    s0, s5 = build().samples[0], build(offset_frames=-5).samples[0]
    assert s5.decision_row == REL - 5
    ys = lambda s: sorted(s.opp_xy[:, 1])
    assert ys(s5)[-1] == pytest.approx(3.0 - 0.08 * (REL - 5)) or any(np.isclose(ys(s5), 3.0 - 0.08 * (REL - 5)))
    with pytest.raises(ValueError):
        DecisionContract(offset_frames=3)                          # decision may not be after release


def test_counts_table_accounts_for_every_candidate():
    r = build(td=make_td(heading=(-1.0, 0.0)))
    c = r.counts()
    assert c.loc["KEPT"].sum() == 0 and c.drop("KEPT").to_numpy().sum() == 1
