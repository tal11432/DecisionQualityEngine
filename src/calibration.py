"""Calibrate and evaluate the pass-feasibility baseline.

Usage: python -m src.calibration --train 1 --test 2

Labels (Metrica event semantics, verified on the data):
  * completed   = PASS events with no subtype (ground passes);
  * intercepted = BALL LOST events with subtype exactly 'INTERCEPTION' (the failed pass itself:
                  From = passer, End X/Y = where the ball was intercepted).
Origin = passer position at the event's start frame; opponents (position + velocity) at that frame.
Target = event end point. CAVEAT: for intercepted passes the end point is where the opponent
actually got the ball, so the target is conditioned on the outcome; the test therefore checks
whether the kinematic model *agrees* that real interceptions were reachable and real completions
were not, using only start-frame information. It is not an out-of-sample forecast of pass outcomes.
Parameters are fitted on the train game only; the test game is touched once, for reporting.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.cli import _load
from src.config import load_config
from src.processing.kinematics import snapshot_velocities, velocity_table
from src.processing.sync import synchronize_events
from src.schema import TrackingData

MIN_PASS_LEN_M = 3.0


# ---- measurements -------------------------------------------------------------------------
def estimate_ball_speed(td: TrackingData, ev: pd.DataFrame, tol: float) -> dict:
    """Average ball speed (chord / duration) of completed ground passes, from tracking ball positions."""
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
    """Quantiles of observed player speeds (central-difference); speeds above ``max_speed`` are tracking artefacts."""
    _, vx, vy = velocity_table(td, half_window)
    sp = np.hypot(vx, vy).ravel()
    sp = sp[np.isfinite(sp) & (sp <= max_speed)]
    return {q: float(np.quantile(sp, q)) for q in quantiles}


# ---- evaluation set -----------------------------------------------------------------------
def build_samples(td: TrackingData, ev: pd.DataFrame, tol: float, half_window: int = 2) -> list[dict]:
    comp = ev[(ev["type"] == "PASS") & ev["subtype"].isna()].assign(label=0)
    icpt = ev[(ev["type"] == "BALL LOST") & (ev["subtype"] == "INTERCEPTION")].assign(label=1)
    sel = pd.concat([comp, icpt])
    matched = synchronize_events(sel, td.frames, tol).matched.set_index("event_idx")
    out = []
    for i, e in sel.iterrows():
        if i not in matched.index or not np.isfinite([e.end_x, e.end_y]).all():
            continue
        row = int(matched.loc[i, "frame_row"])
        snap = td.snapshot(row)
        me = snap.players[snap.players["player_id"] == e.from_player]
        if me.empty:
            continue
        vel = snapshot_velocities(td, row, half_window)
        opp = snap.players[snap.players["team"] != e["team"]].merge(vel, on=["team", "player_id"], how="left")
        origin = (float(me["x"].iloc[0]), float(me["y"].iloc[0]))
        target = (float(e.end_x), float(e.end_y))
        length = float(np.hypot(target[0] - origin[0], target[1] - origin[1]))
        if length < MIN_PASS_LEN_M:
            continue
        out.append({"label": int(e["label"]), "origin": origin, "target": target, "length": length,
                    "opp_xy": opp[["x", "y"]].to_numpy(), "opp_v": opp[["vx", "vy"]].to_numpy()})
    return out


def margins(samples: list[dict], params: FeasibilityParams, use_velocity: bool) -> np.ndarray:
    m = np.array([interception_margin(s["origin"], s["target"], s["opp_xy"], params,
                                      s["opp_v"] if use_velocity else None)["margin_s"] for s in samples])
    finite = m[np.isfinite(m)]
    return np.where(np.isfinite(m), m, (finite.min() - 1) if len(finite) else -1.0)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score_intercepted > score_completed), ties = 0.5 (Mann-Whitney)."""
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(samples: list[dict], params: FeasibilityParams, use_velocity: bool, seed: int = 0, n_boot: int = 500) -> dict:
    y = np.array([s["label"] for s in samples])
    m = margins(samples, params, use_velocity)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        a = auc(m[k], y[k])
        if np.isfinite(a):
            boots.append(a)
    return {"auc": auc(m, y), "auc_ci": tuple(np.percentile(boots, [2.5, 97.5])),
            "flag_rate_intercepted": float((m[y == 1] >= 0).mean()), "flag_rate_completed": float((m[y == 0] >= 0).mean()),
            "n_intercepted": int(y.sum()), "n_completed": int((1 - y).sum())}


def _fmt(name: str, r: dict) -> str:
    return (f"{name:<34} AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  "
            f"flagged: intercepted {r['flag_rate_intercepted']:.1%}, completed {r['flag_rate_completed']:.1%}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--test", type=int, default=2)
    a = ap.parse_args(argv)
    cfg = load_config()
    tol = cfg["sync"]["tolerance_s"]
    td_tr, ev_tr = _load(cfg, a.train)
    td_te, ev_te = _load(cfg, a.test)

    bs = estimate_ball_speed(td_tr, ev_tr, tol)
    qs = estimate_player_speeds(td_tr)
    print(f"[train game {a.train}] ball speed (completed ground passes, n={bs['n']}/{bs['n_candidates']}): "
          f"median {bs['median']:.2f} m/s (IQR {bs['p25']:.2f}-{bs['p75']:.2f})")
    print(f"[train game {a.train}] observed player speed quantiles: " + ", ".join(f"p{int(q * 100)}={v:.2f}" for q, v in qs.items()))

    tr = build_samples(td_tr, ev_tr, tol)
    te = build_samples(td_te, ev_te, tol)
    print(f"samples: train {sum(s['label'] for s in tr)} intercepted / {sum(1 - s['label'] for s in tr)} completed; "
          f"test {sum(s['label'] for s in te)} / {sum(1 - s['label'] for s in te)}")
    print("length-only AUC (is margin better than pass length alone?): "
          f"train {auc(np.array([s['length'] for s in tr]), np.array([s['label'] for s in tr])):.3f}, "
          f"test {auc(np.array([s['length'] for s in te]), np.array([s['label'] for s in te])):.3f}")

    default = FeasibilityParams(15.0, 7.0, 0.3, 60)  # original assumed baseline
    grid = {}
    for (q, v), react in itertools.product(qs.items(), (0.2, 0.3, 0.5)):
        p = FeasibilityParams(bs["median"], v, react, 60)
        grid[(q, react)] = (p, auc(margins(tr, p, True), np.array([s["label"] for s in tr])))
    best_key = max(grid, key=lambda k: grid[k][1])
    best = grid[best_key][0]
    print("\ntrain grid (velocity-aware; ball speed fixed to measured median):")
    for (q, react), (p, au) in grid.items():
        print(f"  player p{int(q * 100)}={p.player_speed:.2f} m/s, reaction {react:.1f}s -> AUC {au:.3f}"
              + ("   <- selected" if (q, react) == best_key else ""))
    print(f"\nSELECTED: ball {best.ball_speed:.2f}, player {best.player_speed:.2f}, reaction {best.reaction_time:.1f}")

    for gname, smp in (("TRAIN", tr), ("TEST (held out)", te)):
        print(f"\n== {gname} ==")
        print(_fmt("original assumed, static", summarize(smp, default, False)))
        print(_fmt("calibrated, static", summarize(smp, best, False)))
        print(_fmt("calibrated + velocities", summarize(smp, best, True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
