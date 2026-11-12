"""Pooled leave-one-game-out validation: correct pooling, train/test isolation, determinism, diagnostics."""
import numpy as np
import pandas as pd
import pytest

import src.pass_completion as pc
import src.pooled_validation as pv
from src.pass_completion_features import FEATURES_A, FEATURES_B


def table(n, seed, sc="A", rate=0.1, slope=1.0):
    r = np.random.default_rng(seed)
    feats = FEATURES_A if sc == "A" else FEATURES_B
    t = pd.DataFrame({f: r.normal(0, 1, n) for f in feats})
    score = t[pv.SCORE[sc]].to_numpy()
    z = np.log((1 - rate) / rate) - slope * score
    t["y_completed"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    t["event_idx"] = np.arange(n)
    return t


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(pv, "N_BOOT", 40)
    monkeypatch.setattr(pv, "N_CAL_BOOT", 10)


def tables(sc="A", sizes=(300, 450, 380)):
    return {1: table(sizes[0], 1, sc, 0.15), 2: table(sizes[1], 2, sc, 0.10, 0.5), 3: table(sizes[2], 3, sc, 0.08, 1.5)}


@pytest.mark.parametrize("sc", ["A", "B"])
def test_folds_pool_exactly_the_other_two_games_and_base_rate_is_pass_weighted(sc):
    T = tables(sc)
    for g in (1, 2, 3):
        r = pv.run_fold(sc, T, g)
        others = [k for k in T if k != g]
        assert r["train"] == others and r["n_train"] == sum(len(T[k]) for k in others) and len(r["te"]) == len(T[g])
        pooled = pd.concat([T[k] for k in others]).y_completed.mean()
        assert r["fit"]["train_rate"] == pytest.approx(pooled)                          # pass-weighted pooled rate
        assert np.allclose(r["P"]["BASE"], pooled)                                      # held-out rate never used
        assert r["train_rates"].keys() == set(others)


def test_unequal_game_sizes_make_pooled_rate_differ_from_mean_of_game_rates():
    T = {1: table(100, 1, "A", 0.4), 2: table(900, 2, "A", 0.05), 3: table(200, 3, "A", 0.1)}
    r = pv.run_fold("A", T, 3)
    mean_of_rates = np.mean([T[1].y_completed.mean(), T[2].y_completed.mean()])
    assert abs(r["fit"]["train_rate"] - mean_of_rates) > 0.03


def test_single_score_and_scenario_separation():
    assert pv.SCORE["A"] in FEATURES_A and pv.SCORE["B"] in FEATURES_B and not set(FEATURES_A) & set(FEATURES_B)
    assert len(pv.SimpleLogit([pv.SCORE["A"]]).fit(table(200, 4, "A")).w) == 1


def test_heldout_labels_and_features_never_reach_deployable_models(monkeypatch):
    T = tables("A")
    base = pv.run_fold("A", T, 3)
    T2 = {k: v.copy() for k, v in T.items()}
    T2[3]["y_completed"] = 1 - T2[3]["y_completed"]                    # flip every held-out label
    T2[3]["best_margin"] = 1e6                                          # and poison a held-out feature
    seen = []
    orig = pc.Prep.fit
    monkeypatch.setattr(pc.Prep, "fit", lambda self, X: (seen.append(float(np.max(X))), orig(self, X))[1])
    flipped = pv.run_fold("A", T2, 3)
    assert max(seen) < 1e5                                              # nothing fitted (scaler, ridge, Platt) saw the held-out game
    probe = T[1].iloc[:60]
    for k in ("M2", "M3", "M3x"):
        np.testing.assert_array_equal(base["fit"][k].predict(probe), flipped["fit"][k].predict(probe))
    assert base["fit"]["train_rate"] == flipped["fit"]["train_rate"]
    # the DIAGNOSTIC oracles are the only quantities allowed to change with held-out labels
    assert not np.allclose(base["diag"]["oracle_intercept"], flipped["diag"]["oracle_intercept"])


def test_scored_predictions_are_invariant_to_heldout_labels():
    """The predictions that are EVALUATED (not just the fitted models) must not depend on held-out labels."""
    T = tables("B")
    base = pv.run_fold("B", T, 1)
    T2 = {k: v.copy() for k, v in T.items()}
    T2[1]["y_completed"] = 1 - T2[1]["y_completed"]                     # labels only; features untouched
    flipped = pv.run_fold("B", T2, 1)
    for k in ("BASE", "M2", "M3", "M3x"):
        np.testing.assert_array_equal(base["P"][k], flipped["P"][k])
    for g in base["single"]:
        np.testing.assert_array_equal(base["single"][g], flipped["single"][g])


def test_meta_pooling_the_heldout_game_would_be_detected(monkeypatch):
    T = tables("A")
    T[3]["best_margin"] = 1e6
    seen = []
    orig = pc.Prep.fit
    monkeypatch.setattr(pc.Prep, "fit", lambda self, X: (seen.append(float(np.max(X))), orig(self, X))[1])
    pv.SimpleLogit(["best_margin"]).fit(pd.concat([T[1], T[3]]))       # the forbidden thing
    assert max(seen) >= 1e5


def test_crossgame_platt_uses_only_training_games_and_never_a_games_own_labels():
    T = tables("A")
    t1, t2 = T[1], T[2]
    p, y = pv.crossgame_oof([t1, t2], ["best_margin"])
    t1b = t1.copy()
    t1b["y_completed"] = 1 - t1b["y_completed"]
    p2, _ = pv.crossgame_oof([t1b, t2], ["best_margin"])
    n1 = len(t1)
    np.testing.assert_array_equal(p[:n1], p2[:n1])                      # game-1 predictions come from a model fitted without game 1 labels
    assert not np.allclose(p[n1:], p2[n1:])                             # game-2 predictions do use game-1 labels (as the model's training data)


def test_deterministic():
    T = tables("B")
    a, b = pv.run_fold("B", T, 2), pv.run_fold("B", T, 2)
    for k in a["P"]:
        np.testing.assert_array_equal(a["P"][k], b["P"][k])
    assert pv.calib_slope_intercept(a["P"]["M3"], a["y"]) == pv.calib_slope_intercept(b["P"]["M3"], b["y"])
    ta, tb = pv.summary_table({2: a}), pv.summary_table({2: b})
    pd.testing.assert_frame_equal(ta, tb)


def test_oracle_diagnostics():
    r = np.random.default_rng(0)
    p = np.clip(r.beta(20, 2, 2000), 0.01, 0.99)
    y = (r.random(2000) < 0.7).astype(int)                              # true rate much lower than predicted
    shifted, delta = pv.oracle_shift(p, y)
    assert shifted.mean() == pytest.approx(y.mean(), abs=1e-6) and delta < 0
    slope, _ = pv.calib_slope_intercept(pv.oracle_recalibrate(p, y), y)
    assert slope == pytest.approx(1.0, abs=0.02)                        # recalibrated on its own labels -> slope 1 (diagnostic by construction)


def test_rate_table_wilson_and_chi_square():
    t = {1: pd.DataFrame({"y_completed": [1] * 90 + [0] * 10}), 2: pd.DataFrame({"y_completed": [1] * 70 + [0] * 30})}
    rt, chi2 = pv.interception_rate_table(t)
    assert chi2 == pytest.approx(12.5)                                  # hand-computed
    assert rt.rate.tolist() == [0.1, 0.3] and (rt.wilson_lo < rt.rate).all() and (rt.rate < rt.wilson_hi).all()
    same = pv.interception_rate_table({1: t[1], 2: t[1]})[1]
    assert same == pytest.approx(0.0)


def test_per_game_relationship_recovers_slope_signs_and_common_fit():
    T = {1: table(1500, 1, "B", 0.1, 0.5), 2: table(1500, 2, "B", 0.1, 1.5), 3: table(1500, 3, "B", 0.05, 1.0)}
    rel = pv.per_game_relationship(T, "margin_static", n_boot=20)
    assert (rel.slope_per_SD < 0).all()
    assert rel.slope_per_SD.iloc[1] < rel.slope_per_SD.iloc[0] < 0         # steeper in game 2
    c = rel.attrs["common"]
    assert c["slope"] < 0 and c["intercepts"]["G3"] > c["intercepts"]["G1"]  # game 3 has the lower interception rate -> higher completion intercept


def test_summary_and_intercept_share_are_bounded():
    r = pv.run_fold("A", tables("A"), 1)
    s = pv.summary_table({1: r})
    assert len(s) == 1 and (np.isnan(s.share_LL_gap_by_intercept.iloc[0]) or -0.5 <= s.share_LL_gap_by_intercept.iloc[0] <= 1.5)
