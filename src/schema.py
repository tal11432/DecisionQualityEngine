"""Internal tracking representation (provider independent).

Canonical coordinates (metres):
  * origin = pitch centre, x along pitch length, y along pitch width, y points "up";
  * x orientation is the AS-RECORDED one of the source (no flipping). Use
    ``TrackingData.snapshot(..., attacking_team=...)`` to get a view in which that
    team attacks towards +x (goal centre at (+length/2, 0)).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pitch:
    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32

    @classmethod
    def from_config(cls, cfg: dict) -> "Pitch":
        p = cfg["pitch"]
        return cls(p["length_m"], p["width_m"], p["goal_width_m"])

    @property
    def goal_center(self) -> tuple[float, float]:
        """Centre of the goal attacked when playing towards +x."""
        return (self.length / 2, 0.0)


@dataclass
class Snapshot:
    """Positions at one instant. ``players``: columns team, player_id, jersey, x, y."""

    players: pd.DataFrame
    ball: tuple[float, float]
    period: int
    frame: int
    time_s: float
    attacking_team: str | None = None  # set if coordinates were oriented

    def team(self, team: str) -> pd.DataFrame:
        return self.players[self.players["team"] == team]


@dataclass
class TrackingData:
    """Wide per-frame table plus player registry.

    frames: columns period, frame, time_s, ball_x, ball_y and, per player,
            ``{team}__{player_id}__x`` / ``__y``.
    players: columns team, player_id, jersey, id_missing (bool).
    attack_dir: (team, period) -> +1 if the team attacks towards +x as recorded, else -1.
    """

    frames: pd.DataFrame
    players: pd.DataFrame
    pitch: Pitch
    source: str = "unknown"
    attack_dir: dict[tuple[str, int], int] = field(default_factory=dict)

    @staticmethod
    def col(team: str, player_id: str, axis: str) -> str:
        return f"{team}__{player_id}__{axis}"

    def player_columns(self) -> list[str]:
        return [c for c in self.frames.columns if c.endswith(("__x", "__y"))]

    def snapshot(self, row: int, attacking_team: str | None = None) -> Snapshot:
        """Snapshot at positional row ``row``. Missing players are omitted (NaN rows dropped)."""
        r = self.frames.iloc[row]
        period = int(r["period"])
        pos = r[self.player_columns() + ["ball_x", "ball_y"]].to_numpy(dtype=float, na_value=np.nan)
        pos = dict(zip(self.player_columns() + ["ball_x", "ball_y"], pos))
        recs = []
        for p in self.players.itertuples():
            x, y = pos[self.col(p.team, p.player_id, "x")], pos[self.col(p.team, p.player_id, "y")]
            if np.isfinite(x) and np.isfinite(y):
                recs.append((p.team, p.player_id, p.jersey, float(x), float(y)))
        df = pd.DataFrame(recs, columns=["team", "player_id", "jersey", "x", "y"])
        ball = (pos["ball_x"], pos["ball_y"])
        if attacking_team is not None:
            d = self.attack_dir.get((attacking_team, period))
            if d is None:
                raise ValueError(f"Attack direction unknown for {attacking_team!r}, period {period}")
            if d == -1:  # rotate 180 degrees so the team attacks +x
                df[["x", "y"]] = -df[["x", "y"]]
                ball = (-ball[0], -ball[1])
        return Snapshot(df, ball, period, int(r["frame"]), float(r["time_s"]), attacking_team)
