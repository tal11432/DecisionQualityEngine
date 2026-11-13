"""DFL/Sportec (IDSSE open dataset) adapter: match-information XML + raw positions XML + raw events XML.

Produces the canonical ``TrackingData`` (metres, centre origin, y up, x as recorded) plus an events table that keeps the
raw provider structure for auditing. It does NOT decide outcome labels (see ``src/idsse_labels.py``).

Design rules (differences from kloppy's reader, which is unsuitable here):
  * a missing/unparseable X or Y is NaN (kloppy would default to 0);
  * the frame index is the UNION of frame numbers over all objects; no frame is dropped because the ball (or any
    player) is absent, and absent players/ball are NaN;
  * time comes from the actual per-frame ``T`` timestamps and the actual ``EventTime`` values on ONE clock
    (seconds since the first tracked frame); no fixed frame offsets are assumed;
  * every tracked person is kept, also if missing from the match information (reported, never dropped);
  * raw event attributes are preserved (flattened JSON in ``raw``, element chain in ``chain``).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.coordinates import infer_attack_direction
from src.schema import Pitch, TrackingData

SECTION_TO_PERIOD = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
SET_PIECES = ("KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty")


@dataclass
class DflMeta:
    match_id: str
    pitch_x: float
    pitch_y: float
    home_id: str
    away_id: str
    kickoff: datetime
    players: dict[str, dict]            # person id -> {team: 'Home'|'Away', shirt, starting}
    team_side: dict[str, str]           # club id -> 'Home' | 'Away'


@dataclass
class TrackingQA:
    """Counts that must be reported, never silently absorbed."""
    n_framesets: int = 0
    n_frames_total: int = 0
    missing_xy: int = 0
    frames_without_ball: int = 0
    t_mismatch_across_objects: int = 0
    persons_not_in_metadata: list[str] = field(default_factory=list)
    metadata_players_never_tracked: list[str] = field(default_factory=list)
    pitch_size_tracking: tuple[float, float] | None = None
    frameset_ranges: dict = field(default_factory=dict)       # (period, person) -> (first N, last N)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def read_match_info(path: str | Path) -> DflMeta:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"match information not found: {path}")
    root = ET.parse(path).getroot()
    gen, env = root.find(".//General"), root.find(".//Environment")
    side = {gen.get("HomeTeamId"): "Home", gen.get("GuestTeamId"): "Away"}
    players: dict[str, dict] = {}
    for team in root.iter("Team"):
        tid = team.get("TeamId")
        if tid not in side:
            raise ValueError(f"unknown team id {tid!r} in match information")
        for p in team.iter("Player"):
            players[p.get("PersonId")] = {"team": side[tid], "shirt": p.get("ShirtNumber"), "starting": p.get("Starting") == "true"}
    return DflMeta(gen.get("MatchId"), float(env.get("PitchX")), float(env.get("PitchY")), gen.get("HomeTeamId"), gen.get("GuestTeamId"),
                   _dt(gen.get("KickoffTime")), players, side)


def _ms(t: str | None) -> int:
    """Timestamp -> integer milliseconds since the epoch (UTC-aware ISO strings); -1 if missing/unparseable."""
    try:
        return int(round(datetime.fromisoformat(t).timestamp() * 1000))
    except (TypeError, ValueError):
        return -1


def _f(v: str | None) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_tracking(positions_xml: str | Path, match_xml: str | Path, pitch: Pitch | None = None) -> tuple[TrackingData, TrackingQA, datetime]:
    """Parse the raw positions file. Returns (TrackingData, QA counters, t0) where ``time_s`` = seconds since ``t0`` (first frame)."""
    positions_xml = Path(positions_xml)
    if not positions_xml.exists():
        raise FileNotFoundError(f"positions file not found: {positions_xml}")
    meta = read_match_info(match_xml)
    pitch = pitch or Pitch(meta.pitch_x, meta.pitch_y)
    if (pitch.length, pitch.width) != (meta.pitch_x, meta.pitch_y):
        raise ValueError(f"pitch {pitch} disagrees with match information {(meta.pitch_x, meta.pitch_y)}")
    qa = TrackingQA()
    sets: dict[tuple[int, str], dict] = {}             # (period, object) -> arrays
    cur = None
    for ev, el in ET.iterparse(str(positions_xml), events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            sec = el.get("GameSection")
            if sec not in SECTION_TO_PERIOD:
                raise ValueError(f"unknown GameSection {sec!r}")
            is_ball = el.get("TeamId") == "BALL"
            obj = "ball" if is_ball else el.get("PersonId")
            cur = {"period": SECTION_TO_PERIOD[sec], "obj": obj, "team": el.get("TeamId"), "N": array("q"), "x": array("d"), "y": array("d"),
                   "z": array("d"), "bs": array("d"), "bp": array("d"), "Tms": array("q")}
            qa.n_framesets += 1
        elif ev == "end" and el.tag == "Frame" and cur is not None:
            n = int(el.get("N"))
            x, y = _f(el.get("X")), _f(el.get("Y"))
            cur["N"].append(n); cur["x"].append(x); cur["y"].append(y)
            cur["Tms"].append(_ms(el.get("T")))
            if cur["obj"] == "ball":
                cur["z"].append(_f(el.get("Z"))); cur["bs"].append(_f(el.get("BallStatus"))); cur["bp"].append(_f(el.get("BallPossession")))
            if x != x or y != y:
                qa.missing_xy += 1
            qa.n_frames_total += 1
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            sets[(cur["period"], cur["obj"])] = cur
            cur = None
            el.clear()
        elif ev == "end" and el.tag == "PitchSize":
            qa.pitch_size_tracking = (float(el.get("X")), float(el.get("Y")))
    if qa.pitch_size_tracking and qa.pitch_size_tracking != (meta.pitch_x, meta.pitch_y):
        raise ValueError(f"pitch size differs between positions {qa.pitch_size_tracking} and match info {(meta.pitch_x, meta.pitch_y)}")

    # master frame index per period = union of frame numbers over ALL objects (nothing is dropped); one clock, T checked across objects
    pidx, tms_by_period = {}, {}
    for p in sorted({k[0] for k in sets}):
        members = [(obj, s) for (pp, obj), s in sets.items() if pp == p]
        union = np.unique(np.concatenate([np.frombuffer(s["N"], dtype=np.int64) for _, s in members]))
        tms = np.full(len(union), -1, dtype=np.int64)
        for obj, s in members:
            pos = np.searchsorted(union, np.frombuffer(s["N"], dtype=np.int64))
            t_obj = np.frombuffer(s["Tms"], dtype=np.int64)
            seen = tms[pos]
            qa.t_mismatch_across_objects += int(((seen != -1) & (seen != t_obj)).sum())
            fill = seen == -1
            tms[pos[fill]] = t_obj[fill]
        if (tms == -1).any():
            raise ValueError(f"{int((tms == -1).sum())} frames in period {p} have no parseable timestamp on any object")
        ball = sets.get((p, "ball"))
        qa.frames_without_ball += len(union) - (len(np.intersect1d(union, np.frombuffer(ball["N"], dtype=np.int64))) if ball else 0)
        pidx[p], tms_by_period[p] = union, tms
    periods_sorted = sorted(pidx)
    period = np.concatenate([np.full(len(pidx[p]), p) for p in periods_sorted])
    frame = np.concatenate([pidx[p] for p in periods_sorted])
    tms_all = np.concatenate([tms_by_period[p] for p in periods_sorted])
    t0 = datetime.fromtimestamp(tms_all[0] / 1000, tz=timezone.utc)
    time_s = (tms_all - tms_all[0]) / 1000.0
    nrows = len(frame)
    offsets, o = {}, 0
    for p in periods_sorted:
        offsets[p] = (o, len(pidx[p])); o += len(pidx[p])

    persons = sorted({obj for (_, obj) in sets if obj != "ball"})
    team_of = {}
    for (_, obj), s in sets.items():
        if obj != "ball":
            team_of[obj] = meta.team_side.get(s["team"])
            if team_of[obj] is None:
                raise ValueError(f"person {obj} has unknown TeamId {s['team']!r}")
    cols = {f"{team_of[o_]}__{o_}__{a}": np.full(nrows, np.nan) for o_ in persons for a in "xy"}
    ball = {k: np.full(nrows, np.nan) for k in ("ball_x", "ball_y", "ball_z", "ball_status", "ball_possession")}
    for (p, obj), s in sets.items():
        off, ln = offsets[p]
        pos = off + np.searchsorted(pidx[p], np.frombuffer(s["N"], dtype=np.int64))
        if obj == "ball":
            ball["ball_x"][pos], ball["ball_y"][pos] = np.frombuffer(s["x"]), np.frombuffer(s["y"])
            ball["ball_z"][pos], ball["ball_status"][pos], ball["ball_possession"][pos] = np.frombuffer(s["z"]), np.frombuffer(s["bs"]), np.frombuffer(s["bp"])
        else:
            cols[f"{team_of[obj]}__{obj}__x"][pos], cols[f"{team_of[obj]}__{obj}__y"][pos] = np.frombuffer(s["x"]), np.frombuffer(s["y"])
        qa.frameset_ranges[(p, obj)] = (int(s["N"][0]), int(s["N"][-1]))
    f = {"period": pd.array(period, dtype="Int64"), "frame": pd.array(frame, dtype="Int64"), "time_s": time_s, **cols, **ball}
    players = pd.DataFrame([(team_of[o_], o_, meta.players.get(o_, {}).get("shirt"), False) for o_ in persons],
                           columns=["team", "player_id", "jersey", "id_missing"])
    qa.persons_not_in_metadata = [o_ for o_ in persons if o_ not in meta.players]
    qa.metadata_players_never_tracked = [o_ for o_ in meta.players if o_ not in persons]
    td = TrackingData(pd.DataFrame(f), players, pitch, source="dfl_idsse")
    team_cols = {t: [td.col(t, p_, "x") for p_ in players.loc[players.team == t, "player_id"]] for t in ("Home", "Away")}
    for per in periods_sorted:
        for team, d in infer_attack_direction(td.frames, team_cols, int(per)).items():
            td.attack_dir[(team, int(per))] = d
    return td, qa, t0


def read_events(events_xml: str | Path, meta: DflMeta, t0: datetime, period_bounds: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """All raw events. ``time_s`` is on the tracking clock (seconds since ``t0``); ``period`` comes from the tracking period time
    ranges (NaN outside them). Coordinates are metres, origin at pitch centre (provider events: bottom-left origin)."""
    events_xml = Path(events_xml)
    if not events_xml.exists():
        raise FileNotFoundError(f"events file not found: {events_xml}")
    root = ET.parse(events_xml).getroot()
    rows = []
    for i, e in enumerate(root.iter("Event")):
        chain, cur = [], e
        flat: dict[str, str] = {}
        while True:
            kids = list(cur)
            if not kids:
                break
            cur = kids[0]
            chain.append(cur.tag)
            for k, v in cur.attrib.items():
                flat[f"{cur.tag}.{k}"] = v
        play = next((c for c in e.iter("Play")), None)
        leaf = next((c for c in (play if play is not None else []) if c.tag in ("Pass", "Cross")), None)
        t = (_dt(e.get("EventTime")) - t0).total_seconds()
        period = next((p for p, (a, b) in period_bounds.items() if a - 1.0 <= t <= b + 1.0), np.nan)
        team_id = (play.get("Team") if play is not None else None) or next((c.get("Team") for c in e.iter() if c.get("Team")), None)
        rows.append({
            "event_idx": i, "event_id": e.get("EventId"), "time_s": t, "period": period, "chain": ">".join(chain),
            "is_delete": "Delete" in chain, "set_piece": next((c for c in chain if c in SET_PIECES), None),
            "is_play": play is not None, "leaf": leaf.tag if leaf is not None else None,
            "direction": leaf.get("Direction") if leaf is not None else None,
            "evaluation": play.get("Evaluation") if play is not None else None,
            "height": play.get("Height") if play is not None else None,
            "team": meta.team_side.get(team_id) if team_id else None,
            "from_player": play.get("Player") if play is not None else next((c.get("Player") for c in e.iter() if c.get("Player")), None),
            "recipient": play.get("Recipient") if play is not None else None,
            "x": _f(e.get("X-Position")) - meta.pitch_x / 2 if e.get("X-Position") else np.nan,
            "y": _f(e.get("Y-Position")) - meta.pitch_y / 2 if e.get("Y-Position") else np.nan,
            "x_src": _f(e.get("X-Source-Position")) - meta.pitch_x / 2 if e.get("X-Source-Position") else np.nan,
            "y_src": _f(e.get("Y-Source-Position")) - meta.pitch_y / 2 if e.get("Y-Source-Position") else np.nan,
            "raw": json.dumps({**{k: v for k, v in e.attrib.items()}, **flat}, ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows)
