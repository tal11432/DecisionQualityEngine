"""Provider-independent football geometry. Inputs are metric (x, y) in a single frame of reference.

Orient coordinates first (``TrackingData.snapshot(attacking_team=...)``) so the goal
attacked is at ``Pitch.goal_center`` = (+L/2, 0). NaN inputs propagate to NaN outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import Snapshot


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    """(n,2) -> (n,n) Euclidean distance matrix."""
    xy = np.asarray(xy, float)
    return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)


def distances_to_point(xy: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    """Distance from each of (n,2) points to ``point`` (ball, goal, ...)."""
    return np.linalg.norm(np.asarray(xy, float) - np.asarray(point, float), axis=-1)


def distance_to_goal(xy: np.ndarray, goal_center: tuple[float, float] = (52.5, 0.0)) -> np.ndarray:
    return distances_to_point(xy, goal_center)


def angle_to_goal(xy: np.ndarray, goal_center: tuple[float, float] = (52.5, 0.0), goal_width: float = 7.32) -> np.ndarray:
    """Angle (radians) subtended at each point by the two goal posts (shooting aperture).

    Goal is assumed to lie on the line x = goal_center[0], posts at y = ±goal_width/2.
    """
    xy = np.asarray(xy, float)
    gx, gy = goal_center
    a = np.stack([gx - xy[..., 0], gy + goal_width / 2 - xy[..., 1]], -1)
    b = np.stack([gx - xy[..., 0], gy - goal_width / 2 - xy[..., 1]], -1)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = (a * b).sum(-1)
    return np.abs(np.arctan2(cross, dot))


def nearest(snap: Snapshot, player_id: str, opponent: bool) -> tuple[str | None, float]:
    """Nearest opponent (``opponent=True``) or teammate of ``player_id``; (None, nan) if none tracked."""
    p = snap.players
    me = p[p["player_id"] == player_id]
    if me.empty:
        raise KeyError(f"player {player_id!r} not tracked in this snapshot")
    me = me.iloc[0]
    pool = p[(p["team"] != me["team"]) if opponent else ((p["team"] == me["team"]) & (p["player_id"] != player_id))]
    if pool.empty:
        return None, float("nan")
    d = np.hypot(pool["x"] - me["x"], pool["y"] - me["y"])
    i = d.idxmin()
    return str(pool.loc[i, "player_id"]), float(d.loc[i])


def nearest_defender(snap: Snapshot, player_id: str) -> tuple[str | None, float]:
    return nearest(snap, player_id, opponent=True)


def nearest_teammate(snap: Snapshot, player_id: str) -> tuple[str | None, float]:
    return nearest(snap, player_id, opponent=False)


def point_segment_distance(points: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """Distance from each of (n,2) points to the segment a-b."""
    pts, a_, b_ = np.asarray(points, float), np.asarray(a, float), np.asarray(b, float)
    ab = b_ - a_
    denom = float(ab @ ab)
    t = np.zeros(len(pts)) if denom == 0 else np.clip(((pts - a_) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a_ + t[:, None] * ab), axis=-1)


def passing_lane_obstruction(passer: tuple[float, float], receiver: tuple[float, float],
                             defenders: np.ndarray, lane_width: float = 2.0) -> dict:
    """Geometric obstruction check: defenders within ``lane_width``/2 metres of the pass segment.

    This is a static-geometry heuristic. It ignores ball speed, defender motion and
    reaction time, so it is NOT a pass-feasibility or interception-probability model.
    """
    d = np.asarray(defenders, float).reshape(-1, 2)
    d = d[np.isfinite(d).all(axis=1)]
    if len(d) == 0:
        return {"obstructed": False, "n_blockers": 0, "min_distance": float("nan")}
    dist = point_segment_distance(d, passer, receiver)
    n = int((dist <= lane_width / 2).sum())
    return {"obstructed": n > 0, "n_blockers": n, "min_distance": float(dist.min())}


def snapshot_features(snap: Snapshot, goal_center: tuple[float, float] = (52.5, 0.0), goal_width: float = 7.32) -> pd.DataFrame:
    """Per-player table: distance to ball / goal, angle to goal, nearest opponent & teammate."""
    p = snap.players.copy()
    xy = p[["x", "y"]].to_numpy(float)
    p["dist_ball"] = distances_to_point(xy, snap.ball)
    p["dist_goal"] = distance_to_goal(xy, goal_center)
    p["angle_goal_rad"] = angle_to_goal(xy, goal_center, goal_width)
    d = pairwise_distances(xy)
    np.fill_diagonal(d, np.inf)  # exclude self
    team = p["team"].to_numpy()
    same = team[:, None] == team[None, :]
    p["nearest_opp_dist"] = np.where(~same, d, np.inf).min(1) if len(p) else []
    p["nearest_mate_dist"] = np.where(same, d, np.inf).min(1) if len(p) else []
    return p.replace([np.inf], np.nan)
