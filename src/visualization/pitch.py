"""Pitch and frame plotting (canonical metric coordinates)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

from src.schema import Pitch, Snapshot, TrackingData

TEAM_COLORS = {"Home": "#d62728", "Away": "#1f77b4"}


def draw_pitch(ax, pitch: Pitch = Pitch()) -> None:
    L, W = pitch.length, pitch.width
    kw = dict(fill=False, ec="#444", lw=1.2)
    ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, **kw))
    ax.plot([0, 0], [-W / 2, W / 2], c="#444", lw=1.2)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * L / 2 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((s * L / 2 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, **kw))
        ax.add_patch(Rectangle((s * L / 2 - (0 if s > 0 else 2), -pitch.goal_width / 2), 2, pitch.goal_width, fill=False, ec="#444", lw=1.5))
        ax.add_patch(Arc((s * (L / 2 - 11), 0), 18.3, 18.3, theta1=-53 if s > 0 else 127, theta2=53 if s > 0 else 233, ec="#444", lw=1.2))
    ax.set_xlim(-L / 2 - 4, L / 2 + 4)
    ax.set_ylim(-W / 2 - 4, W / 2 + 4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_snapshot(snap: Snapshot, pitch: Pitch = Pitch(), ax=None, event: dict | None = None, label: str = "jersey"):
    """Plot one snapshot. ``label``: 'jersey' | 'id' | 'none'. ``event``: optional dict with
    start_x/start_y/end_x/end_y/type/team/from_player/to_player in the SAME frame of reference."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, pitch)
    for team, g in snap.players.groupby("team"):
        ax.scatter(g["x"], g["y"], s=220, c=TEAM_COLORS.get(team, "gray"), ec="white", zorder=3, label=team)
        if label != "none":
            for r in g.itertuples():
                txt = (r.jersey if label == "jersey" and r.jersey else r.player_id.replace("Player", ""))
                ax.text(r.x, r.y, txt, ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=4)
    if np.isfinite(snap.ball).all():
        ax.scatter(*snap.ball, s=70, c="black", marker="o", ec="yellow", zorder=5, label="ball")
    else:
        ax.text(0, -pitch.width / 2 - 2.5, "ball not tracked", ha="center", color="gray")
    title = f"period {snap.period}  frame {snap.frame}  t={snap.time_s:.2f}s"
    if snap.attacking_team:
        title += f"  (oriented: {snap.attacking_team} attacks →)"
    if event is not None:
        title += f"\n{event.get('team')} {event.get('type')} {event.get('from_player')}→{event.get('to_player')}"
        sx, sy, ex, ey = (event.get(k) for k in ("start_x", "start_y", "end_x", "end_y"))
        if np.isfinite([sx, sy]).all():
            ax.scatter(sx, sy, s=380, facecolors="none", ec="green", lw=2, zorder=6)
            if np.isfinite([ex, ey]).all():
                ax.annotate("", (ex, ey), (sx, sy), arrowprops=dict(arrowstyle="->", color="green", lw=2), zorder=6)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def _orient_event(event: dict, flip: bool) -> dict:
    e = dict(event)
    if flip:
        for k in ("start_x", "start_y", "end_x", "end_y"):
            e[k] = -e[k]
    return e


def plot_frame(td: TrackingData, row: int, attacking_team: str | None = None, **kw):
    """Plot tracking row ``row`` (positional index into ``td.frames``)."""
    return plot_snapshot(td.snapshot(row, attacking_team), td.pitch, **kw)


def plot_event(td: TrackingData, event: dict, frame_row: int, attacking_team: str | None = None, **kw):
    """Plot an event over its synchronized tracking frame (event coords follow the same orientation)."""
    snap = td.snapshot(frame_row, attacking_team)
    flip = bool(attacking_team and td.attack_dir[(attacking_team, snap.period)] == -1)
    return plot_snapshot(snap, td.pitch, event=_orient_event(event, flip), **kw)
