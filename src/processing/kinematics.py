"""Velocity estimation from adjacent frames (central difference)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import TrackingData


def velocity_table(td: TrackingData, half_window: int = 2) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Central-difference velocities (m/s) for every player and frame.

    v[i] = (pos[i+h] - pos[i-h]) / (t[i+h] - t[i-h]), valid only if both frames exist in the
    same period and are exactly 2h frame numbers apart (no gaps); otherwise NaN.
    Returns (keys [(team, player_id)], vx, vy) with vx, vy shaped (n_frames, n_players).
    """
    h = half_window
    if h < 1:
        raise ValueError("half_window must be >= 1")
    f = td.frames
    keys = list(td.players[["team", "player_id"]].itertuples(index=False, name=None))
    x = np.column_stack([f[td.col(t, p, "x")].to_numpy(float) for t, p in keys])
    y = np.column_stack([f[td.col(t, p, "y")].to_numpy(float) for t, p in keys])
    t = f["time_s"].to_numpy(float)
    fr = f["frame"].to_numpy(dtype=float, na_value=np.nan)
    per = f["period"].to_numpy(dtype=float, na_value=np.nan)
    n = len(f)
    vx, vy = np.full_like(x, np.nan), np.full_like(y, np.nan)
    if n > 2 * h:
        lo, hi = slice(0, n - 2 * h), slice(2 * h, n)
        ok = (per[lo] == per[hi]) & (fr[hi] - fr[lo] == 2 * h) & (t[hi] - t[lo] > 0)
        dt = np.where(ok, t[hi] - t[lo], np.nan)[:, None]
        mid = slice(h, n - h)
        vx[mid] = (x[hi] - x[lo]) / dt
        vy[mid] = (y[hi] - y[lo]) / dt
    return keys, vx, vy


def snapshot_velocities(td: TrackingData, row: int, half_window: int = 2) -> pd.DataFrame:
    """Velocities at one row: columns team, player_id, vx, vy (NaN where unavailable)."""
    h = half_window
    f = td.frames
    out = []
    ok = row - h >= 0 and row + h < len(f)
    if ok:
        i0, i1 = row - h, row + h
        ok = bool(f["period"].iloc[i0] == f["period"].iloc[i1]
                  and f["frame"].iloc[i1] - f["frame"].iloc[i0] == 2 * h
                  and f["time_s"].iloc[i1] > f["time_s"].iloc[i0])
        dt = float(f["time_s"].iloc[i1] - f["time_s"].iloc[i0]) if ok else np.nan
    for team, pid in td.players[["team", "player_id"]].itertuples(index=False, name=None):
        if ok:
            vx, vy = ((float(f[td.col(team, pid, ax)].iloc[i1]) - float(f[td.col(team, pid, ax)].iloc[i0])) / dt
                      for ax in ("x", "y"))
        else:
            vx = vy = np.nan
        out.append((team, pid, vx, vy))
    return pd.DataFrame(out, columns=["team", "player_id", "vx", "vy"])
