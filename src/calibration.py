"""Calibrate (development game) and evaluate (held-out game) the pass-feasibility score, leakage-safe.

Usage: python -m src.calibration --dev 1 --test 2 [--out reports/pass_feasibility_results.txt]

Everything fitted or selected here (ball-speed prior, speed grid, F1 threshold) uses the DEVELOPMENT
game only. The held-out game is scored once with frozen settings. Sample construction follows the
decision-time contract in ``src/pass_dataset.py``.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.cli import _load
from src.config import load_config
from src.audit import endpoint_target_margins
from src.evaluation import auc, auc_ci, best_f1_threshold, boot_indices, clean, confusion, paired_auc_diff, subgroup_auc
from src.pass_dataset import DecisionContract, DecisionSample, build_decision_samples
from src.processing.kinematics import velocity_table
from src.processing.sync import synchronize_events
from src.schema import TrackingData

# ---- measurements -------------------------------------------------------------------------
def estimate_ball_speed(td: TrackingData, ev: pd.DataFrame, tol: float) -> dict:
    """Population prior: median average ball speed (chord / duration) of completed ground passes.

    Uses the passes' end frames, so it is an outcome-derived POPULATION statistic from the
    development game. It is applied as a fixed constant to every decision; no per-pass eventual
    speed ever enters a prediction."""
    p = ev[(ev["type"] == "PASS") & ev["subtype"].isna()]
    s = synchronize_events(p, td.frames, tol, "start_time_s", "start_frame").matched.set_index("event_idx")
    e = synchronize_events(p, td.frames, tol, "end_time_s", "end_frame").matched.set_index("event_idx")
    idx = s.index.intersection(e.index)
    f = td.frames
    a = f[["ball_x", "ball_y"]].to_numpy(float)[s.loc[idx, "frame_row"].to_numpy()]
    b = f[["ball_x", "ball_y"]].to_numpy(float)[e.loc[idx, "frame_row"].to_numpy()]
    dt = f["time_s"].to_numpy(float)[e.loc[idx, "frame_row"].to_numpy()] - f["time_s"].to_numpy(float)[s.loc[idx, "frame_row"].to_numpy()]
    dist = np.linalg.norm(b - a, axis=1)
    ok = np.isfinite(dist) & (dist >= 5.0) & (dt >= 0.3) & (dt <= 5.0)
    v = dist[ok] / dt[ok]
    return {"n": int(ok.sum()), "n_candidates": len(idx), "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}


def estimate_player_speeds(td: TrackingData, quantiles=(0.90, 0.95, 0.99), max_speed: float = 12.0,
                           half_window: int = 2) -> dict[float, float]:
    """OFFLINE population speed quantiles (central difference); speeds above ``max_speed`` are artefacts.
    These are quantiles over all player-frames (mostly walking/jogging): descriptive, not a top-speed measurement."""
    _, vx, vy = velocity_table(td, half_window)
    sp = np.hypot(vx, vy).ravel()
    sp = sp[np.isfinite(sp) & (sp <= max_speed)]
    return {q: float(np.quantile(sp, q)) for q in quantiles}


@dataclass(frozen=True)
class Model:
    name: str
    params: FeasibilityParams
    use_velocity: bool


def score_margin(samples: list[DecisionSample], model: Model) -> np.ndarray:
    return clean(np.array([interception_margin(s.origin, s.target, s.opp_xy, model.params,
                                               s.opp_v if model.use_velocity else None)["margin_s"] for s in samples]))


def _y(samples): return np.array([s.label for s in samples])


def _ci(sc, y, idx):
    a, lo, hi = auc_ci(sc, y, idx)
    return f"{a:.3f} [{lo:.3f}, {hi:.3f}]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=int, default=1)
    ap.add_argument("--test", type=int, default=2)
    ap.add_argument("--out", default=None, help="also write the report text here")
    a = ap.parse_args(argv)
    cfg = load_config()
    tol = cfg["sync"]["tolerance_s"]
    contract = DecisionContract.from_config(cfg)
    L: list[str] = []
    say = L.append

    td_d, ev_d = _load(cfg, a.dev)
    td_t, ev_t = _load(cfg, a.test)
    say(f"DEV game {a.dev} (calibration/selection)  |  TEST game {a.test} (held out, scored once)")
    say(f"contract: {contract}")

    # ---- dev-only measurements ----
    bs = estimate_ball_speed(td_d, ev_d, tol)
    qs = estimate_player_speeds(td_d)
    say(f"\n[dev] ball-speed prior: median {bs['median']:.2f} m/s (IQR {bs['p25']:.2f}-{bs['p75']:.2f}, n={bs['n']}/{bs['n_candidates']} completed ground passes)")
    say("[dev] observed player-speed quantiles over all player-frames: " + ", ".join(f"p{int(q * 100)}={v:.2f}" for q, v in qs.items()))

    # ---- samples & accounting ----
    dev, test = build_decision_samples(td_d, ev_d, tol, contract), build_decision_samples(td_t, ev_t, tol, contract)
    for name, r in (("DEV", dev), ("TEST", test)):
        say(f"\n[{name}] sample accounting (first failing check; rows are exclusive)")
        say(r.counts().to_string())
        asg = r.assignments
        c = asg[(asg.label == 0) & asg.true_receiver.notna()]
        got = c[c.assigned.notna()]
        say(f"[{name}] target-identification rule on completed passes with known receiver: assigned {len(got)}/{len(c)}, "
            f"correct {int((got.assigned == got.true_receiver).sum())}/{len(got)} "
            f"({(got.assigned == got.true_receiver).mean():.1%}). Intercepted-pass accuracy cannot be measured (no receiver in data).")
    ytr, yte = _y(dev.samples), _y(test.samples)
    if len(dev.samples) == 0 or len(test.samples) == 0 or yte.sum() == 0:
        say("\nINSUFFICIENT DATA for evaluation")
        print("\n".join(L))
        return 1

    # ---- models ----
    o = FeasibilityParams(15.0, 7.0, 0.3, 60)
    m1 = Model("M1 original static", o, False)
    m2 = Model("M2 calibrated ball speed", FeasibilityParams(bs["median"], 7.0, 0.3, 60), False)
    m3 = Model("M3 M2 + causal velocities", m2.params, True)
    grid = {}
    for v, react in itertools.product([7.0, *qs.values()], (0.2, 0.3, 0.5)):
        mm = Model("grid", FeasibilityParams(bs["median"], v, react, 60), True)
        grid[(v, react)] = auc(score_margin(dev.samples, mm), ytr)
    gk = max(grid, key=grid.get)
    m4 = Model(f"M4 dev-refit (v={gk[0]:.2f}, rt={gk[1]:.1f}) + causal vel", FeasibilityParams(bs["median"], gk[0], gk[1], 60), True)
    say("\n[dev] player-speed / reaction grid (ball speed fixed; velocities on); dev AUC (diagnostic; weakly identified):")
    for (v, react), au in grid.items():
        say(f"  speed {v:.2f} m/s, reaction {react:.1f}s -> {au:.3f}" + ("   <- M4 (selected on dev)" if (v, react) == gk else ""))
    models = [m1, m2, m3, m4]

    def all_scores(r):
        sc = {m.name: score_margin(r.samples, m) for m in models}
        sc["Length only"] = np.array([s.length for s in r.samples])
        return sc

    sd, st = all_scores(dev), all_scores(test)
    idx_d, idx_t = boot_indices(len(ytr)), boot_indices(len(yte))
    for name, sc, y, idx in (("DEV", sd, ytr, idx_d), ("TEST (held out)", st, yte, idx_t)):
        say(f"\n== {name}: n={len(y)} ({int(y.sum())} intercepted / {int((1 - y).sum())} completed; base rate {y.mean():.1%}) ==")
        say("AUC [95% bootstrap CI over decisions, 2000 resamples]:")
        for k, s in sc.items():
            say(f"  {k:<48} {_ci(s, y, idx)}")

    say("\n== TEST: paired AUC differences on identical samples (95% CI; significant = CI excludes 0) ==")
    pairs = [("M2 calibrated ball speed", "M1 original static"), ("M3 M2 + causal velocities", "M2 calibrated ball speed"),
             (m4.name, "M3 M2 + causal velocities"), ("M1 original static", "Length only"),
             ("M3 M2 + causal velocities", "Length only")]
    for p, q in pairs:
        d = paired_auc_diff(st[p], st[q], yte, idx_t)
        say(f"  {p} - {q}: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] {'SIGNIFICANT' if d['significant'] else 'not significant'}")

    say("\n== TEST: confusion (positive = intercepted). The score is a geometric margin (s), NOT a probability. ==")
    say("threshold A = margin >= 0 (the model's own 'interceptable' definition; fixed a priori); "
        "threshold B = F1-optimal on DEV, applied unchanged to TEST")
    for k in st:
        for tag, thr in (("A", 0.0), ("B", best_f1_threshold(sd[k], ytr))):
            if k == "Length only" and tag == "A":
                continue
            c = confusion(st[k], yte, thr)
            say(f"  {k:<48} {tag} thr={c['threshold']:+8.2f}  TP {c['tp']:>3} FP {c['fp']:>3} FN {c['fn']:>3} TN {c['tn']:>4}  "
                f"P {c['precision']:.3f} R {c['recall']:.3f} F1 {c['f1']:.3f}")

    say("\n== TEST: subgroups (AUC [CI]; 'n/a' if either class < 15) ==")
    lens = np.array([s.length for s in test.samples])
    for title, g in (("pass length", pd.Series(pd.cut(lens, [0, 10, 20, 200], labels=["3-10 m", "10-20 m", ">=20 m"], right=False).astype(str))),
                     ("team", pd.Series([s.team for s in test.samples])),
                     ("period", pd.Series([f"period {s.period}" for s in test.samples]))):
        say(f"-- by {title}")
        say(subgroup_auc({k: st[k] for k in ("M1 original static", "M3 M2 + causal velocities", "Length only")}, yte, g).to_string(index=False))

    say("\n== LEAK SIZE: same decisions, target replaced by the ACTUAL end point (discarded legacy protocol) ==")
    for name, r, ev, y, idx in (("DEV", dev, ev_d, ytr, idx_d), ("TEST", test, ev_t, yte, idx_t)):
        leaky = endpoint_target_margins(r.samples, ev, o)
        ok = np.isfinite(leaky)
        sub = idx  # same resampling; unusable rows scored at sentinel
        say(f"  {name}: M1 AUC with actual-endpoint target {_ci(clean(leaky), y, sub)}  vs leakage-safe target "
            f"{_ci(st['M1 original static'] if name == 'TEST' else sd['M1 original static'], y, sub)}  (end point missing for {int((~ok).sum())})")

    say("\n== DEV-ONLY sensitivity: decision frame 5 frames (0.2 s) BEFORE release ==")
    c5 = DecisionContract(**{**contract.__dict__, "offset_frames": -5})
    d5 = build_decision_samples(td_d, ev_d, tol, c5)
    y5 = _y(d5.samples)
    say(f"  n={len(y5)} ({int(y5.sum())} intercepted); M1 {_ci(score_margin(d5.samples, m1), y5, boot_indices(len(y5)))}; "
        f"M3 {_ci(score_margin(d5.samples, m3), y5, boot_indices(len(y5)))}")
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
