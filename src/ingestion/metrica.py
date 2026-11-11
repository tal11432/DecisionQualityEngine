"""Metrica Sports sample-data adapter (standard CSV format, Sample Games 1 & 2).

Tracking CSV layout (one file per team, 25 Hz):
  row 0: team label per player column (``Home``/``Away``)
  row 1: jersey numbers
  row 2: ``Period, Frame, Time [s], <PlayerId>, '', ..., Ball, ''``  (x,y column pairs)
  row 3+: data; coordinates normalized to [0,1], NaN = not tracked.
Events CSV: Team, Type, Subtype, Period, Start Frame, Start Time [s], End Frame,
End Time [s], From, To, Start X, Start Y, End X, End Y.
Sample Game 3 (EPTS/FIFA format) is NOT supported by this adapter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.coordinates import infer_attack_direction, metrica_to_metric
from src.schema import Pitch, TrackingData

_N_HEADER = 3


def read_team_file(path: str | Path, team: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one team file -> (frames in metres, player registry). Includes the ball."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")
    head = pd.read_csv(path, header=None, nrows=_N_HEADER, dtype=str, keep_default_na=False)
    if len(head) < _N_HEADER:
        raise ValueError(f"{path.name}: expected {_N_HEADER} header rows")
    labels, jerseys, names = (head.iloc[i].tolist() for i in range(3))
    if [n.strip() for n in names[:3]] != ["Period", "Frame", "Time [s]"]:
        raise ValueError(f"{path.name}: unexpected header {names[:3]}, not Metrica CSV format")
    ncol = len(names)
    if (ncol - 3) % 2 or names[-2].strip() != "Ball":
        raise ValueError(f"{path.name}: expected x/y column pairs ending with 'Ball'")
    file_team = {l.strip() for l in labels[3:] if l.strip()}
    if file_team != {team}:
        raise ValueError(f"{path.name}: header team labels {sorted(file_team)} != expected {team!r}")

    raw = pd.read_csv(path, header=None, skiprows=_N_HEADER, na_values=["NaN", "nan", ""])
    if raw.shape[1] != ncol:
        raise ValueError(f"{path.name}: {raw.shape[1]} data columns but {ncol} header columns")
    raw = raw.apply(pd.to_numeric, errors="coerce")

    out = {"period": raw[0].astype("Int64"), "frame": raw[1].astype("Int64"), "time_s": raw[2].astype(float)}
    # Pitch is applied by the caller; here keep normalized values, converted later.
    reg = []
    for i in range(3, ncol - 2, 2):
        pid = names[i].strip()
        missing = pid == ""
        if missing:
            pid = f"UNKNOWN_{i}"
        jersey = jerseys[i].strip() or None
        reg.append((team, pid, jersey, missing))
        out[TrackingData.col(team, pid, "x")] = raw[i]
        out[TrackingData.col(team, pid, "y")] = raw[i + 1]
    out["ball_x"], out["ball_y"] = raw[ncol - 2], raw[ncol - 1]
    players = pd.DataFrame(reg, columns=["team", "player_id", "jersey", "id_missing"])
    if players.duplicated(["team", "player_id"]).any():
        raise ValueError(f"{path.name}: duplicate player identifiers in header")
    return pd.DataFrame(out), players


def read_tracking(home_csv: str | Path, away_csv: str | Path, pitch: Pitch | None = None) -> TrackingData:
    """Load Home+Away files into a normalized :class:`TrackingData` (metres, as-recorded orientation).

    Both files must carry identical period/frame/time columns, otherwise ValueError (no silent merge).
    Duplicate or missing frames are preserved and reported by ``processing.validation``.
    """
    pitch = pitch or Pitch()
    h, hp = read_team_file(home_csv, "Home")
    a, ap = read_team_file(away_csv, "Away")
    keys = ["period", "frame", "time_s"]
    if len(h) != len(a) or not h[keys].equals(a[keys]):
        raise ValueError("Home and Away tracking files are not aligned on period/frame/time")
    frames = pd.concat([h, a.drop(columns=keys + ["ball_x", "ball_y"])], axis=1)
    # Ball is stored in both files; they must agree, otherwise we refuse to guess.
    if not np.allclose(h[["ball_x", "ball_y"]], a[["ball_x", "ball_y"]], equal_nan=True):
        raise ValueError("Ball coordinates differ between Home and Away files")

    # normalized -> metres
    for c in frames.columns:
        if c.endswith("__x") or c == "ball_x":
            frames[c] = metrica_to_metric(frames[c], 0.0, pitch)[0]
        elif c.endswith("__y") or c == "ball_y":
            frames[c] = metrica_to_metric(0.0, frames[c], pitch)[1]
    players = pd.concat([hp, ap], ignore_index=True)
    td = TrackingData(frames, players, pitch, source="metrica_csv")

    team_cols = {t: [td.col(t, p, "x") for p in players.loc[players.team == t, "player_id"]] for t in ("Home", "Away")}
    for period in sorted(frames["period"].dropna().unique()):
        for team, d in infer_attack_direction(frames, team_cols, int(period)).items():
            td.attack_dir[(team, int(period))] = d
    return td


def read_events(path: str | Path, pitch: Pitch | None = None) -> pd.DataFrame:
    """Load Metrica events; snake_case columns, coordinates converted to canonical metres."""
    pitch = pitch or Pitch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")
    ev = pd.read_csv(path)
    expected = ["Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]", "End Frame",
                "End Time [s]", "From", "To", "Start X", "Start Y", "End X", "End Y"]
    miss = [c for c in expected if c not in ev.columns]
    if miss:
        raise ValueError(f"{path.name}: missing event columns {miss}")
    ev = ev[expected].rename(columns={
        "Team": "team", "Type": "type", "Subtype": "subtype", "Period": "period",
        "Start Frame": "start_frame", "Start Time [s]": "start_time_s", "End Frame": "end_frame",
        "End Time [s]": "end_time_s", "From": "from_player", "To": "to_player",
        "Start X": "start_x", "Start Y": "start_y", "End X": "end_x", "End Y": "end_y"})
    ev["start_x"], ev["start_y"] = metrica_to_metric(ev["start_x"], ev["start_y"], pitch)
    ev["end_x"], ev["end_y"] = metrica_to_metric(ev["end_x"], ev["end_y"], pitch)
    return ev
