"""Kinematic pass-feasibility baseline (time-to-intercept). Provider independent.

Model: the ball travels the straight pass line at constant speed. An opponent can
intercept at a lane point if ``reaction + distance / player_speed <= ball_time`` there.
Opponents are static unless velocities are supplied (then they drift at their current
velocity during the reaction time); there is no acceleration model. It is a deterministic
baseline, not a probability of completion; parameters should come from ``src/calibration.py``. It is the geometric prerequisite for alternative-action generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.schema import Snapshot


@dataclass(frozen=True)
class FeasibilityParams:
    ball_speed: float = 15.0
    player_speed: float = 7.0
    reaction_time: float = 0.3
    lane_samples: int = 60

    @classmethod
    def from_config(cls, cfg: dict) -> "FeasibilityParams":
        c = cfg["feasibility"]
        return cls(c["ball_speed_ms"], c["player_speed_ms"], c["reaction_time_s"], c["lane_samples"])

    def __post_init__(self):
        if min(self.ball_speed, self.player_speed) <= 0 or self.reaction_time < 0 or self.lane_samples < 2:
            raise ValueError("invalid feasibility parameters")


def interception_margin(origin, target, opponents: np.ndarray, params: FeasibilityParams = FeasibilityParams(),
                        opp_velocities: np.ndarray | None = None) -> dict:
    """Earliest-threat margin for a pass origin->target.

    margin_s = min over lane points and opponents of (ball_time - opponent_time).
    margin_s >= 0 means some opponent can reach a lane point no later than the ball
    (interceptable); more negative = safer. NaN opponents are ignored; with none, margin is +inf-safe
    (returned as ``-inf``) and ``interceptable`` is False. Also returns the intercepting opponent index.

    ``opp_velocities`` (m/s, same shape as ``opponents``) are optional. If given, each opponent is
    first advanced by ``velocity * reaction_time`` (they keep their current motion until they react);
    NaN velocities are treated as zero (static). Without it, opponents are static.
    """
    o, t = np.asarray(origin, float), np.asarray(target, float)
    opp = np.asarray(opponents, float).reshape(-1, 2)
    keep = np.isfinite(opp).all(axis=1)
    opp = opp[keep]
    if opp_velocities is not None:
        vel = np.nan_to_num(np.asarray(opp_velocities, float).reshape(-1, 2)[keep])
        opp = opp + vel * params.reaction_time
    length = float(np.linalg.norm(t - o))
    if len(opp) == 0:
        return {"interceptable": False, "margin_s": -np.inf, "pass_length": length, "threat_idx": None}
    s = np.linspace(0.0, 1.0, params.lane_samples)[1:]          # exclude the origin itself
    pts = o + s[:, None] * (t - o)                              # (k,2)
    t_ball = s * length / params.ball_speed                     # (k,)
    d = np.linalg.norm(opp[:, None, :] - pts[None, :, :], axis=-1)   # (m,k)
    t_opp = params.reaction_time + d / params.player_speed
    margin = t_ball[None, :] - t_opp                            # >=0: opponent arrives in time
    per_opp = margin.max(axis=1)
    i = int(per_opp.argmax())
    return {"interceptable": bool(per_opp[i] >= 0), "margin_s": float(per_opp[i]),
            "pass_length": length, "threat_idx": i}


def receiver_arrival_margin(origin, target, receiver_pos, params: FeasibilityParams = FeasibilityParams()) -> float:
    """ball_time - receiver_time at ``target`` (>= 0 means the receiver gets there in time)."""
    o, t, r = (np.asarray(v, float) for v in (origin, target, receiver_pos))
    return float(np.linalg.norm(t - o) / params.ball_speed - np.linalg.norm(t - r) / params.player_speed)


def pass_options(snap: Snapshot, passer_id: str, params: FeasibilityParams = FeasibilityParams()) -> pd.DataFrame:
    """Feasibility of a pass from ``passer_id`` to each tracked teammate's current position.

    Columns: receiver, pass_length, interceptable, margin_s, threat_player. Receivers sit on
    the target, so receiver arrival is trivially satisfied; use ``receiver_arrival_margin``
    for passes into space.
    """
    p = snap.players
    me = p[p["player_id"] == passer_id]
    if me.empty:
        raise KeyError(f"player {passer_id!r} not tracked in this snapshot")
    me = me.iloc[0]
    opp = p[p["team"] != me["team"]].reset_index(drop=True)
    rows = []
    vel = opp[["vx", "vy"]].to_numpy() if {"vx", "vy"} <= set(opp.columns) else None
    for r in p[(p["team"] == me["team"]) & (p["player_id"] != passer_id)].itertuples():
        m = interception_margin((me["x"], me["y"]), (r.x, r.y), opp[["x", "y"]].to_numpy(), params, vel)
        threat = None if m["threat_idx"] is None else str(opp.loc[m["threat_idx"], "player_id"])
        rows.append((r.player_id, m["pass_length"], m["interceptable"], m["margin_s"], threat))
    return pd.DataFrame(rows, columns=["receiver", "pass_length", "interceptable", "margin_s", "threat_player"])
