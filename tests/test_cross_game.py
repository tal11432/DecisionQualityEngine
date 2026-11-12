"""Cross-game evaluation: train-only fitting, identical test samples, determinism, validation gate, opt-in repair."""
import numpy as np
import pandas as pd
import pytest

import src.cross_game as cg
import src.pass_completion_features as pcf
from src.pass_completion_features import FEATURES_A


def table(n, seed, shift=0.0):
    r = np.random.default_rng(seed)
    t = pd.DataFrame({f: r.normal(shift, 1, n) for f in FEATURES_A})
    p = 1 / (1 + np.exp(-(1.8 + 0.8 * t["n_safe_options"] - 0.6 * t["best_margin"])))
    t["y_completed"] = (r.random(n) < p).astype(int)
    t["event_idx"] = np.arange(n)
    return t


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(cg.pc, "ROUNDS_GRID", (10, 25))
    monkeypatch.setattr(cg, "N_BOOT", 40)


def test_fit_uses_training_table_only_and_run_is_deterministic():
    tr, te = table(250, 1), table(180, 2)
    f1, f2 = cg.fit_train("A", tr), cg.fit_train("A", tr)
    probe = tr.iloc[:50]
    for n in f1["models"]:
        np.testing.assert_array_equal(f1["models"][n].predict(probe), f2["models"][n].predict(probe))
    te_marked = te.copy()
    te_marked.loc[:, "dist_goal"] = 1e6                                   # only the test table carries this
    f3 = cg.fit_train("A", tr)                                            # fit is a function of the training table alone
    np.testing.assert_array_equal(f1["models"][f1["pname"]].predict(probe), f3["models"][f3["pname"]].predict(probe))
    r1, r2 = cg.evaluate_run("A", f1, te), cg.evaluate_run("A", f1, te)
    assert r1["metrics"]["prim"]["roc_auc"] == r2["metrics"]["prim"]["roc_auc"]
    cg.evaluate_run("A", f1, te_marked)                                   # scoring a different test table never refits
    np.testing.assert_array_equal(f1["models"][f1["pname"]].predict(probe), f3["models"][f3["pname"]].predict(probe))


def test_all_models_scored_on_identical_test_samples_and_paired_diffs_present():
    tr, te = table(250, 3), table(180, 4)
    r = cg.evaluate_run("A", cg.fit_train("A", tr), te)
    assert r["n"] == 180 and all(len(p) == 180 for p in r["P"].values())
    assert set(r["vs_base"]) == {"b1", "b2", "prim", "gbt"} and set(r["feas_vs_dist"]) == {"roc_auc", "log_loss"}
    assert r["n_int"] == int((1 - te.y_completed).sum())


def test_baseline_is_the_training_base_rate_not_the_test_rate():
    tr, te = table(250, 5), table(180, 6, shift=0.0)
    te["y_completed"] = 1                                                  # a very different test base rate
    f = cg.fit_train("A", tr)
    p = f["models"]["Base rate"].predict(te)
    assert np.allclose(p, tr.y_completed.mean()) and not np.allclose(p, te.y_completed.mean())


def test_verdict_rules():
    d = lambda diff, sig: {"diff": diff, "significant": sig}
    assert cg.verdict({"brier": d(-0.01, True), "log_loss": d(-0.02, True)}, (0.6, 0.55, 0.65)) == "prob.skill:yes, ranking:yes"
    assert cg.verdict({"brier": d(-0.01, False), "log_loss": d(-0.02, True)}, (0.6, 0.45, 0.65)) == "prob.skill:partial, ranking:no"
    assert cg.verdict({"brier": d(+0.01, True), "log_loss": d(+0.02, True)}, (0.6, 0.55, 0.65)).startswith("prob.skill:no")   # significantly WORSE is not skill


def test_evaluation_reuses_the_existing_feature_builder_unchanged():
    assert cg.build_features is pcf.build_features and cg.pc.SCENARIO["A"]["sets"]["A6 full"] == FEATURES_A


def test_failed_gate_blocks_all_model_evaluation(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cg, "_load", lambda cfg, g: (None, None))
    monkeypatch.setattr(cg, "gate", lambda cfg, loaded, say: False)

    def boom(*a, **k):
        raise AssertionError("features must not be built when the gate fails")
    monkeypatch.setattr(cg, "build_features", boom)
    out = tmp_path / "r.txt"
    assert cg.main(["--out", str(out)]) == 1
    assert "Model evaluation NOT run" in out.read_text()


def test_gate_fails_on_too_few_labelled_interceptions(monkeypatch):
    from tests.test_leakage import events, make_td
    td = make_td()
    td.attack_dir = {("Home", 1): 1, ("Away", 1): -1}
    ev = events()                                                          # a single completed pass: no interceptions at all
    lines = []
    assert cg.gate({"sync": {"tolerance_s": 0.02}, "validation": {}}, {1: (td, ev)}, lines.append) is False
    assert "FAIL" in lines[1]


def test_identity_fix_is_opt_in():
    from tests.test_epts import _swap_case
    td, ev = _swap_case(swap=True)
    assert cg.maybe_fix_identity({3: (td, ev)}, False)[3][0] is td
    fixed = cg.maybe_fix_identity({3: (td, ev)}, True)[3][0]
    assert fixed is not td and "identity_swaps_applied" in fixed.source and fixed.frames["Home__H1__x"].iloc[0] == 20.0
