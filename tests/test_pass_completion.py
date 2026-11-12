"""Leakage protection, label rules, train/test separation and reproducibility for pass completion."""
import numpy as np
import pandas as pd
import pytest

import src.pass_completion as pc
from src import pass_completion_evaluation as pe
from src.analytics.feasibility import FeasibilityParams
from src.pass_completion_features import (ALLOWED_EVENT_COLUMNS, FEATURES, FEATURES_A, FEATURES_B, build_features,
                                          label_events)
from src.pass_dataset import DecisionContract
from tests.test_leakage import K_FRAMES, N, REL, TOL, events, make_td

P = FeasibilityParams(11.9, 7.0, 0.3, 60)
C = DecisionContract()


def td_():
    t = make_td()
    t.attack_dir = {("Home", 1): 1, ("Away", 1): -1}
    return t


def feats(td, ev, sc):
    return build_features(td, ev, sc, TOL, C, P)


# ------------------------------------------------------------------ labels
def _ev(rows):
    cols = ["team", "type", "subtype", "period", "start_time_s", "end_time_s", "from_player", "to_player"]
    return pd.DataFrame(rows, columns=cols)


def test_label_rules_and_unknowns():
    ev = _ev([
        ("Home", "PASS", None, 1, 1.0, 2.0, "H1", "H2"),                  # 0 completed
        ("Home", "PASS", "HEAD", 1, 1.0, 2.0, "H1", "H2"),                # 1 unknown: subtype
        ("Home", "PASS", None, 1, 1.0, 2.0, "H1", None),                  # 2 unknown: no receiver
        ("Home", "PASS", None, 1, 1.0, 1.0, "H1", "H2"),                  # 3 unknown: zero duration
        ("Home", "BALL LOST", "INTERCEPTION", 1, 1.0, 2.0, "H1", None),   # 4 intercepted
        ("Home", "BALL LOST", "INTERCEPTION", 1, 1.0, 2.0, None, None),   # 5 unknown: no passer
        ("Home", "BALL LOST", "THEFT", 1, 1.0, 2.0, "H1", None),          # 6 unknown: tackle, NOT interception
        ("Home", "BALL LOST", None, 1, 1.0, 2.0, "H1", None),             # 7 unknown
        ("Home", "BALL LOST", "CROSS-INTERCEPTION", 1, 1.0, 2.0, "H1", None),  # 8 unknown: non-ground
        ("Away", "RECOVERY", "INTERCEPTION", 1, 1.0, 1.0, "A1", None),    # 9 not pass-related: not listed
    ])
    lab = label_events(ev).set_index("event_idx")
    assert lab.outcome.dropna().to_dict() == {0: "completed", 4: "intercepted"}
    assert set(lab.index[lab.outcome.isna()]) == {1, 2, 3, 5, 6, 7, 8}
    assert 9 not in lab.index
    assert all(lab.loc[i, "reason"].startswith("unknown") for i in (1, 2, 3, 5, 6, 7, 8))


def test_unknown_outcomes_are_counted_not_featurised():
    td = td_()
    rows = [events(), events("BALL LOST", to=None), events("BALL LOST", to=None)]
    ev = pd.concat(rows, ignore_index=True)
    ev.loc[2, "subtype"] = "THEFT"                       # a possession loss that must not become an interception
    fs = feats(td, ev, "A")
    assert len(fs.unknown) == 1 and "THEFT" in fs.unknown.reason.iloc[0]
    assert sorted(fs.table.event_idx) == [0, 1] and fs.table.y_completed.tolist() == [1, 0]


# ------------------------------------------------------------------ registry / contract
def test_event_column_whitelist_has_no_outcome_columns():
    banned = {"end_x", "end_y", "end_time_s", "end_frame", "to_player", "start_x", "start_y", "start_frame", "type", "subtype"}
    assert not banned & set(ALLOWED_EVENT_COLUMNS)


def test_feature_registry_documents_every_feature_and_scenarios_are_disjoint():
    assert set(FEATURES_A).isdisjoint(FEATURES_B)
    for name, (scn, unit, definition, when) in FEATURES.items():
        assert scn in "AB" and unit and definition and when
        assert "end point" not in definition and "interception location" not in definition
    # scenario A is decision-time: every timestamp is <= t_d; scenario B is labelled early flight
    assert all(FEATURES[f][3].startswith("t_d") for f in FEATURES_A)
    assert any("t_p" in FEATURES[f][3] for f in FEATURES_B)


def test_tables_contain_only_their_own_scenario_features_and_frames():
    td = td_()
    a, b = feats(td, events(), "A").table, feats(td, events(), "B").table
    assert set(FEATURES_A) <= set(a.columns) and not set(FEATURES_B) & set(a.columns)
    assert set(FEATURES_B) <= set(b.columns) and not set(FEATURES_A) & set(b.columns)
    assert a.row.iloc[0] == REL and b.row.iloc[0] == REL + K_FRAMES      # t_d vs t_p = t_d + 6 frames


# ------------------------------------------------------------------ invariance harness (+ meta-tests)
def scramble_outcomes(ev):
    ev = ev.copy()
    for c in ("end_x", "end_y", "start_x", "start_y"):
        ev[c] = 999.0
    ev["end_time_s"] = ev["start_time_s"] + 5.0          # still a valid positive duration (labels unchanged)
    ev["end_frame"], ev["start_frame"] = 12345, 12345
    ev["to_player"] = "A2"
    return ev


def scramble_future(td, from_row):
    td = td_()
    rng = np.random.default_rng(0)
    for c in [c for c in td.frames.columns if c.endswith(("__x", "__y"))] + ["ball_x", "ball_y"]:
        td.frames.loc[from_row:, c] = rng.normal(0, 30, N - from_row)
    return td


def assert_invariant(builder, scenario):
    """builder(td, events, scenario) -> feature DataFrame must ignore outcome columns and frames after t."""
    base = builder(td_(), events(), scenario)
    pd.testing.assert_frame_equal(base, builder(td_(), scramble_outcomes(events()), scenario))
    last_used = REL if scenario == "A" else REL + K_FRAMES
    pd.testing.assert_frame_equal(base, builder(scramble_future(td_(), last_used + 1), events(), scenario))


def real_builder(td, ev, sc):
    return feats(td, ev, sc).table


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_features_ignore_outcomes_and_future_frames(scenario):
    assert_invariant(real_builder, scenario)


def test_meta_harness_detects_endpoint_leak():
    def leaky(td, ev, sc):                               # injected leak: pass length from the actual end point
        t = real_builder(td, ev, sc).copy()
        t["leak"] = np.hypot(ev["end_x"].iloc[0], ev["end_y"].iloc[0])
        return t
    with pytest.raises(AssertionError):
        assert_invariant(leaky, "A")


def test_meta_harness_detects_future_frame_leak():
    def leaky(td, ev, sc):                               # injected leak: velocity from a frame AFTER the prediction frame
        t = real_builder(td, ev, sc).copy()
        t["leak"] = td.frames["Away__A1__y"].iloc[REL + K_FRAMES + 3]
        return t
    with pytest.raises(AssertionError):
        assert_invariant(leaky, "B")


def test_future_receiver_movement_cannot_enter_scenario_b():
    """Target teammate's positions after t_p are scrambled: receiver features must not move."""
    base = feats(td_(), events(), "B").table
    td = td_()
    for ax in ("x", "y"):
        td.frames.loc[REL + K_FRAMES + 1:, td.col("Home", "H2", ax)] = 500.0
    pd.testing.assert_frame_equal(base, feats(td, events(), "B").table)


# ------------------------------------------------------------------ target inference
def test_scenario_b_target_comes_from_heading_not_from_receiver_or_endpoint():
    ev = scramble_outcomes(events())                     # to_player = A2, end point = (999, 999)
    row = feats(td_(), ev, "B").table.iloc[0]
    td = td_()
    ball0 = td.frames.loc[REL, ["ball_x", "ball_y"]].to_numpy(float)
    h2 = td.frames.loc[REL + K_FRAMES, ["Home__H2__x", "Home__H2__y"]].to_numpy(float)
    assert row.pass_dist == pytest.approx(np.hypot(*(h2 - ball0)))


def test_unidentified_target_is_excluded_even_if_endpoint_sits_on_a_teammate():
    td = make_td(heading=(0.6, -0.8))
    td.attack_dir = {("Home", 1): 1, ("Away", 1): -1}
    fs = feats(td, events("BALL LOST", end=(15.0, 0.5), to="H2"), "B")
    assert fs.table.empty and fs.exclusions.reason.iloc[0].startswith("no teammate within heading cone")
    assert len(feats(td, events("BALL LOST", end=(15.0, 0.5), to="H2"), "A").table) == 1   # A needs no target


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_completed_and_intercepted_share_feature_construction(scenario):
    td = td_()
    a = feats(td, events("PASS", to="H2"), scenario).table
    b = feats(td, events("BALL LOST", end=(70.0, 70.0), to=None), scenario).table
    assert a.y_completed.iloc[0] == 1 and b.y_completed.iloc[0] == 0
    pd.testing.assert_frame_equal(a.drop(columns="y_completed"), b.drop(columns="y_completed"))


def test_decision_time_target_not_identifiable_flag_is_scenario_a_only():
    # scenario A never calls the heading rule: a pass into space still yields an A sample
    td = make_td(heading=(-1.0, 0.0))
    td.attack_dir = {("Home", 1): 1, ("Away", 1): -1}
    assert len(feats(td, events(), "A").table) == 1 and feats(td, events(), "B").table.empty


# ------------------------------------------------------------------ models & numerics
def synth(n=400, seed=0, shift=0.0):
    r = np.random.default_rng(seed)
    X = r.normal(shift, 1, (n, 3))
    p = 1 / (1 + np.exp(-(1.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1])))
    y = (r.random(n) < p).astype(int)
    return pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "y_completed": y})


def test_logistic_recovers_coefficients_and_ridge_shrinks():
    r = np.random.default_rng(1)
    Z = r.normal(size=(5000, 2))
    y = (r.random(5000) < 1 / (1 + np.exp(-(0.5 + 1.0 * Z[:, 0] - 0.5 * Z[:, 1])))).astype(float)
    w, b = pc.fit_logistic(Z, y, 1e-6)
    assert w == pytest.approx([1.0, -0.5], abs=0.1) and b == pytest.approx(0.5, abs=0.1)
    w2, _ = pc.fit_logistic(Z, y, 1e4)
    assert np.abs(w2).sum() < np.abs(w).sum() * 0.5


def test_boosted_trees_learn_an_interaction_deterministically():
    r = np.random.default_rng(2)
    X = r.uniform(-1, 1, (600, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)          # XOR: not learnable by a linear model
    m = pc.BoostedTrees(100).fit(X, y)
    acc = ((m.predict(X) > 0.5) == y).mean()
    assert acc > 0.9
    np.testing.assert_array_equal(m.predict(X), pc.BoostedTrees(100).fit(X, y).predict(X))
    lin_w, lin_b = pc.fit_logistic(X, y, 1.0)
    assert (((pc.sigmoid(X @ lin_w + lin_b)) > 0.5) == y).mean() < 0.7


def test_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2])
    assert pe.roc_auc(p, y) == 1.0 and pe.pr_auc_intercepted(p, y) == 1.0
    assert pe.brier(p, y) == pytest.approx(np.mean([0.01, 0.04, 0.09, 0.04]))
    assert pe.log_loss(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(np.log(2))
    assert pe.pr_auc_intercepted(np.array([0.9, 0.8, 0.3, 0.2]), np.array([0, 0, 1, 1])) == pytest.approx((1 / 3 + 2 / 4) / 2)   # positives ranked 3rd and 4th
    assert pe.threshold_f1(p, y) == pytest.approx(0.3)           # flag rule is p <= tau: flags exactly the two intercepted passes


def test_calibration_bins_respect_sample_size():
    r = np.random.default_rng(3)
    y = (r.random(300) < 0.9).astype(int)
    c = pe.calibration_assessment(np.full(300, 0.9), y, pc.fit_logistic)
    assert c["n_bins"] == min(5, int((1 - y).sum()) // 10)
    few = pe.calibration_assessment(np.full(50, 0.9), np.r_[np.ones(46), np.zeros(4)].astype(int), pc.fit_logistic)
    assert few["n_bins"] < 2 and few["table"].empty


def test_block_ids_are_contiguous_and_balanced():
    b = pc.block_ids(103, 5)
    assert (np.diff(b) >= 0).all() and set(b) == set(range(5))


def test_one_se_rule_prefers_simpler_set_when_indistinguishable():
    df = synth(600, 4)
    df["noise"] = np.random.default_rng(9).normal(size=len(df))
    sets = {"S1": ["f1", "f2"], "S2": ["f1", "f2", "f3", "noise"]}
    chosen, _, tab = pc.select_feature_set(df, sets)
    assert chosen == "S1"
    assert pc.select_feature_set(df, {"only noise": ["noise", "f3"], "signal": ["f1", "f2"]})[0] == "signal"


# ------------------------------------------------------------------ train/test separation
def test_fitted_model_independent_of_test_data_and_never_sees_it(monkeypatch):
    train, test = synth(300, 5), synth(200, 6, shift=3.0)
    m1 = pc.fit_model(train, "m", ["f1", "f2"], "logit", 10.0)
    m2 = pc.fit_model(train, "m", ["f1", "f2"], "logit", 10.0)
    assert (m1.platt, m1.tau) == (m2.platt, m2.tau)
    np.testing.assert_array_equal(m1.coef, m2.coef)
    # nothing fitted (scaler, ridge, Platt, threshold) may have seen marker values that only exist in the test set
    test_marked = test.copy()
    test_marked.loc[:, "f1"] = 1e6
    seen = []
    orig = pc.Prep.fit

    def spy(self, X):
        seen.append(float(np.max(X)))
        return orig(self, X)
    monkeypatch.setattr(pc.Prep, "fit", spy)
    pc.fit_model(train, "m", ["f1", "f2"], "logit", 10.0).predict(test_marked)
    assert seen and max(seen) < 1e5


def test_meta_pooling_train_and_test_is_detected(monkeypatch):
    train, test = synth(300, 5), synth(200, 6)
    test.loc[:, "f1"] = 1e6
    seen = []
    orig = pc.Prep.fit
    monkeypatch.setattr(pc.Prep, "fit", lambda self, X: (seen.append(float(np.max(X))), orig(self, X))[1])
    pc.fit_model(pd.concat([train, test], ignore_index=True), "m", ["f1", "f2"], "logit", 10.0)   # the forbidden thing
    assert max(seen) >= 1e5                              # i.e. the check in the previous test WOULD have failed


def test_out_of_fold_prediction_does_not_use_its_own_block():
    df = synth(250, 7)
    X, y = df[["f1", "f2"]].to_numpy(), df.y_completed.to_numpy()
    base = pc.oof_predict(X, y, pc.logit_fp(10.0))
    b = pc.block_ids(len(y))
    y2 = y.copy()
    y2[b == 2] = 1 - y2[b == 2]                          # flip labels of block 2 only
    flipped = pc.oof_predict(X, y2, pc.logit_fp(10.0))
    np.testing.assert_array_equal(base[b == 2], flipped[b == 2])      # block-2 predictions never saw block-2 labels
    assert not np.allclose(base[b != 2], flipped[b != 2])


# ------------------------------------------------------------------ reproducibility
def test_run_scenario_is_deterministic_and_reports_both_class_counts(monkeypatch):
    monkeypatch.setattr(pc, "N_BOOT", 40)
    monkeypatch.setattr(pc, "N_COEF_BOOT", 10)
    monkeypatch.setattr(pc, "ROUNDS_GRID", (10, 25))
    tr, te = synth(200, 8), synth(150, 9)
    for t in (tr, te):
        for f in FEATURES_A:
            t[f] = np.random.default_rng(len(f)).normal(size=len(t)) + t.y_completed * 0.3
        t["event_idx"] = np.arange(len(t))
    out = []
    for _ in range(2):
        lines = []
        pc.run_scenario("A", tr, te, None, lines.append)
        out.append("\n".join(lines))
    assert out[0] == out[1] and "intercepted" in out[0] and "SCENARIO A" in out[0]
