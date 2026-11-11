"""Coordinate conversion and attacking-direction inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import Pitch


def metrica_to_metric(x, y, pitch: Pitch):
    """Metrica normalized coords -> canonical metres.

    Metrica: (0,0) top-left, (1,1) bottom-right, y points DOWN. Canonical: origin at
    centre, y points UP, so ``y_m`` is flipped. Works on scalars or arrays.
    """
    return (np.asarray(x, dtype=float) - 0.5) * pitch.length, (0.5 - np.asarray(y, dtype=float)) * pitch.width


def infer_attack_direction(frames: pd.DataFrame, team_cols: dict[str, list[str]], period: int) -> dict[str, int]:
    """Infer attacking direction per team from the first frame of ``period`` with data.

    At kick-off each team stands in its own half, so the sign of the team's median x
    is its defended side. Raises if the result is ambiguous (teams not on opposite sides).

    team_cols: team -> list of x-column names.
    """
    sub = frames[frames["period"] == period]
    if sub.empty:
        raise ValueError(f"No frames for period {period}")
    row_idx = None
    for i in range(len(sub)):
        if all(sub.iloc[i][cols].notna().sum() >= 5 for cols in team_cols.values()):
            row_idx = i
            break
    if row_idx is None:
        raise ValueError(f"No frame in period {period} with >=5 tracked players per team")
    row = sub.iloc[row_idx]
    med = {t: float(np.nanmedian(row[cols].to_numpy(dtype=float))) for t, cols in team_cols.items()}
    signs = {t: int(np.sign(m)) for t, m in med.items()}
    if len(set(signs.values())) != len(signs) or 0 in signs.values():
        raise ValueError(f"Ambiguous attack direction in period {period}: median x by team = {med}")
    return {t: -s for t, s in signs.items()}  # defends side s -> attacks the other way
