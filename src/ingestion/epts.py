"""EPTS/FIFA adapter (Metrica Sample Game 3). Produces the same TrackingData / events schema as ``metrica.py``.

Differences from the Metrica CSV format (games 1-2), all handled here:
  * tracking.txt: one line per frame ``frame:x,y;x,y;...(22 players):ballx,bally`` (no header, NaN = untracked);
    the player slots are POSITIONAL and change at every substitution, so slot -> player comes from the
    ``DataFormatSpecification`` segment (startFrame..endFrame) of metadata.xml. A player is NaN while off the pitch.
  * metadata.xml gives frame rate, field size, period frame ranges, team sides, shirt numbers, attack direction in half 1.
  * events.json: ids instead of names (team FIFATMA/B, player Pxxxx), ``subtypes`` is null/dict/list (joined with '-'
    to match the CSV vocabulary, e.g. 'HEAD-INTERCEPTION'), start/end carry frame, time, x, y (same normalized space).
  * Time: provider ``time`` = frame / fps (global clock across periods), same as the CSV games.
Assumptions (verified on Game 3: event positions equal ball positions at the event frame, unflipped, normalized 0-1, y down):
  * coordinates use the same normalization/orientation as the CSV games -> ``metrica_to_metric``;
  * FIFATMA (local team) -> 'Home', FIFATMB (visiting) -> 'Away'; player ids are the provider ids (e.g. 'P3578').
Nothing is invented: untracked values stay NaN; frames with no roster/period, a pitch-size mismatch or an attack
direction that contradicts the metadata raise ValueError.
"""
from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.coordinates import infer_attack_direction, metrica_to_metric
from src.schema import Pitch, TrackingData


@dataclass
class EptsMeta:
    fps: float
    pitch_size: tuple[float, float]                     # (length, width) m
    periods: dict[int, tuple[int, int]]                 # period -> (first frame, last frame)
    team_side: dict[str, str]                           # provider team id -> 'Home' | 'Away'
    attack_first_half: dict[str, str]                   # 'Home'|'Away' -> 'left_to_right' | 'right_to_left'
    shirt: dict[str, str]                               # player id -> shirt number
    player_team: dict[str, str]                         # player id -> 'Home' | 'Away'
    segments: list[tuple[int, int, list[str]]] = field(default_factory=list)   # (start, end, ordered player ids)


def read_metadata(path: str | Path) -> EptsMeta:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    root = ET.parse(path).getroot()
    prm = {p.findtext("Name"): p.findtext("Value") for p in root.find("Metadata/GlobalConfig").iter("ProviderParameter")}
    periods = {}
    for i, (a, b) in enumerate((("first_half_start", "first_half_end"), ("second_half_start", "second_half_end"),
                                ("first_extra_half_start", "first_extra_half_end"), ("second_extra_half_start", "second_extra_half_end")), 1):
        if prm.get(a) and prm.get(b):
            periods[i] = (int(prm[a]), int(prm[b]))
    score = root.find("Metadata/Sessions/Session/MatchParameters/Score")
    side = {score.get("idLocalTeam"): "Home", score.get("idVisitingTeam"): "Away"}
    attack = {}
    for t in root.iter("Team"):
        for p in t.iter("ProviderParameter"):
            if p.findtext("Name") == "attack_direction_first_half":
                attack[side[t.get("id")]] = p.findtext("Value")
    fs = root.find("Metadata/Sessions/Session/FieldSize")
    players = {p.get("id"): p for p in root.iter("Player")}
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    segs = []
    for s in root.iter("DataFormatSpecification"):
        if s.get("separator") != ":":
            raise ValueError("unsupported tracking layout (top-level separator != ':')")
        ids = [chan[r.get("playerChannelId")] for r in s.iter("PlayerChannelRef") if r.get("playerChannelId").endswith("_x")]
        segs.append((int(s.get("startFrame")), int(s.get("endFrame")), ids))
    return EptsMeta(float(root.findtext("Metadata/GlobalConfig/FrameRate")), (float(fs.findtext("Width")), float(fs.findtext("Height"))),
                    periods, side, attack,
                    {i: p.findtext("ShirtNumber") for i, p in players.items()}, {i: side[p.get("teamId")] for i, p in players.items()}, segs)


def read_tracking(tracking_txt: str | Path, metadata_xml: str | Path, pitch: Pitch | None = None) -> TrackingData:
    """Load EPTS tracking into the canonical wide TrackingData (metres, as-recorded orientation)."""
    pitch = pitch or Pitch()
    tracking_txt = Path(tracking_txt)
    if not tracking_txt.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_txt}")
    meta = read_metadata(metadata_xml)
    if not np.allclose(meta.pitch_size, (pitch.length, pitch.width)):
        raise ValueError(f"pitch size mismatch: metadata {meta.pitch_size} vs configured {(pitch.length, pitch.width)}")
    n_slots = {len(ids) for _, _, ids in meta.segments}
    if len(n_slots) != 1:
        raise ValueError(f"segments disagree on the number of player slots: {sorted(n_slots)}")
    n_slots = n_slots.pop()
    text = tracking_txt.read_text().replace(":", ",").replace(";", ",")
    # pandas silently NaN-pads short lines, which would turn a malformed line into "untracked" data: check explicitly
    want = 2 * n_slots + 2
    bad = [i + 1 for i, line in enumerate(text.splitlines()) if line.count(",") != want]
    if bad:
        raise ValueError(f"{tracking_txt.name}: {len(bad)} malformed line(s) (wrong field count), first at line {bad[0]}")
    try:
        raw = pd.read_csv(io.StringIO(text), header=None, na_values=["NaN", "nan", ""], dtype=float)
    except Exception as e:  # ragged lines etc.
        raise ValueError(f"cannot parse {tracking_txt.name}: {e}") from e
    if raw.shape[1] != 1 + 2 * n_slots + 2:
        raise ValueError(f"{tracking_txt.name}: {raw.shape[1]} columns, expected {1 + 2 * n_slots + 2}")
    frame = raw[0].to_numpy()
    n = len(frame)
    roster = [(meta.player_team[p], p) for p in dict.fromkeys(p for _, _, ids in meta.segments for p in ids)]
    cols = {TrackingData.col(t, p, a): np.full(n, np.nan) for t, p in roster for a in "xy"}
    covered = np.zeros(n, bool)
    for s, e, ids in meta.segments:
        m = (frame >= s) & (frame <= e)
        covered |= m
        for k, p in enumerate(ids):
            t = meta.player_team[p]
            cols[TrackingData.col(t, p, "x")][m] = raw[1 + 2 * k].to_numpy()[m]
            cols[TrackingData.col(t, p, "y")][m] = raw[2 + 2 * k].to_numpy()[m]
    if not covered.all():
        raise ValueError(f"{int((~covered).sum())} frames fall outside every DataFormatSpecification segment (player identity unknown)")
    period = np.full(n, np.nan)
    for per, (s, e) in meta.periods.items():
        period[(frame >= s) & (frame <= e)] = per
    if np.isnan(period).any():
        raise ValueError(f"{int(np.isnan(period).sum())} frames outside every period range in metadata")
    f = {"period": pd.array(period, dtype="Int64"), "frame": pd.array(frame, dtype="Int64"), "time_s": frame / meta.fps}
    for c, v in cols.items():
        f[c] = metrica_to_metric(v, 0.0, pitch)[0] if c.endswith("__x") else metrica_to_metric(0.0, v, pitch)[1]
    f["ball_x"], f["ball_y"] = metrica_to_metric(raw[1 + 2 * n_slots].to_numpy(), raw[2 + 2 * n_slots].to_numpy(), pitch)
    players = pd.DataFrame([(t, p, meta.shirt[p], False) for t, p in roster], columns=["team", "player_id", "jersey", "id_missing"])
    td = TrackingData(pd.DataFrame(f), players, pitch, source="epts")
    team_cols = {t: [td.col(t, p, "x") for p in players.loc[players.team == t, "player_id"]] for t in ("Home", "Away")}
    for per in sorted(meta.periods):
        for team, d in infer_attack_direction(td.frames, team_cols, per).items():
            td.attack_dir[(team, per)] = d
    # cross-check the kick-off inference against the provider's stated half-1 direction (+1 = towards +x = left_to_right)
    for team, word in meta.attack_first_half.items():
        if td.attack_dir[(team, 1)] != (1 if word == "left_to_right" else -1):
            raise ValueError(f"inferred attack direction for {team} contradicts metadata ({word})")
    return td


def read_events(events_json: str | Path, metadata_xml: str | Path, pitch: Pitch | None = None) -> pd.DataFrame:
    """Events in the same schema as ``metrica.read_events`` (snake_case, canonical metres, Home/Away teams)."""
    pitch = pitch or Pitch()
    events_json = Path(events_json)
    if not events_json.exists():
        raise FileNotFoundError(f"Events file not found: {events_json}")
    meta = read_metadata(metadata_xml)
    rows = []
    for e in json.loads(events_json.read_text())["data"]:
        sub = e.get("subtypes")
        sub = None if sub is None else ("-".join(d["name"] for d in sub) if isinstance(sub, list) else sub["name"])
        if e["team"]["id"] not in meta.team_side:
            raise ValueError(f"unknown team id {e['team']['id']!r} in events")
        s, t = e["start"], e["end"]
        rows.append((meta.team_side[e["team"]["id"]], e["type"]["name"], sub, e["period"], s["frame"], s["time"], t["frame"], t["time"],
                     (e.get("from") or {}).get("id"), (e.get("to") or {}).get("id"),
                     s["x"], s["y"], t["x"], t["y"]))
    ev = pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start_frame", "start_time_s", "end_frame", "end_time_s",
                                     "from_player", "to_player", "start_x", "start_y", "end_x", "end_y"])
    for c in ("start_x", "start_y", "end_x", "end_y"):
        ev[c] = ev[c].astype(float)
    ev["start_x"], ev["start_y"] = metrica_to_metric(ev["start_x"], ev["start_y"], pitch)
    ev["end_x"], ev["end_y"] = metrica_to_metric(ev["end_x"], ev["end_y"], pitch)
    return ev


def check_events(events: pd.DataFrame, td: TrackingData, fps: float = 25.0, tol_s: float = 0.02) -> dict:
    """Consistency of events with tracking: unknown player ids, wrong-team players, events whose time != frame / fps."""
    team_of = dict(zip(td.players.player_id, td.players.team))
    known = events["from_player"].dropna()
    return {"events": len(events),
            "from_player_unknown": int((~known.isin(team_of)).sum()),
            "from_player_wrong_team": int(sum(team_of.get(p) not in (None, t) for p, t in zip(events["from_player"], events["team"]) if pd.notna(p))),
            "start_clock_mismatch": int((np.abs(events["start_time_s"] - events["start_frame"] / fps) > tol_s).sum())}


# ---------------------------------------------------------------- identity consistency (event passer vs ball position)
def detect_identity_swaps(td: TrackingData, events: pd.DataFrame, tol_s: float = 0.02, min_events: int = 10,
                          min_rate: float = 0.9) -> list[dict]:
    """Detect pairs of same-team players whose tracking slots look swapped relative to the events.

    For each event the player nearest to the ball at the synchronised start frame is compared with the named passer.
    A swap (a, b, period) is reported only if >= ``min_rate`` of a's events (>= ``min_events``) have the ball at b AND
    vice versa: systematic and mutual, never a one-off. Uses only passer/ball proximity (no outcome information).
    """
    from src.processing.sync import synchronize_events
    ev = events[events["from_player"].notna()]
    m = synchronize_events(ev[["team", "period", "start_time_s", "from_player"]], td.frames, tol_s, "start_time_s", None).matched
    f = td.frames
    keys = list(td.players[["team", "player_id"]].itertuples(index=False, name=None))
    X = np.column_stack([f[td.col(t, p, "x")].to_numpy(float) for t, p in keys])
    Y = np.column_stack([f[td.col(t, p, "y")].to_numpy(float) for t, p in keys])
    tally: dict[tuple, dict] = {}
    for _, r in m.iterrows():
        row, e = int(r["frame_row"]), ev.loc[r["event_idx"]]
        bx, by = f["ball_x"].iloc[row], f["ball_y"].iloc[row]
        if np.isnan(bx):
            continue
        d = np.hypot(X[row] - bx, Y[row] - by)
        if np.isnan(d).all():
            continue
        near = keys[int(np.nanargmin(d))]
        c = tally.setdefault((e["team"], int(e["period"]), e["from_player"]), {})
        c[near[1]] = c.get(near[1], 0) + 1
    out = []
    for (team, per, a), ca in tally.items():
        n_a = sum(ca.values())
        if n_a < min_events:
            continue
        b = max(ca, key=ca.get)
        if b == a or ca[b] / n_a < min_rate:
            continue
        cb = tally.get((team, per, b), {})
        n_b = sum(cb.values())
        if a < b and n_b >= min_events and cb.get(a, 0) / n_b >= min_rate:
            out.append({"team": team, "period": per, "a": a, "b": b, "n_a": n_a, "n_b": n_b,
                        "rate_a_at_b": ca[b] / n_a, "rate_b_at_a": cb[a] / n_b})
    return out


def apply_identity_swaps(td: TrackingData, swaps: list[dict]) -> TrackingData:
    """Return a copy of ``td`` with the detected pairs' tracking columns exchanged for the affected period (opt-in repair)."""
    f = td.frames.copy()
    for s in swaps:
        rows = (f["period"] == s["period"]).to_numpy()
        for ax in "xy":
            ca, cb = td.col(s["team"], s["a"], ax), td.col(s["team"], s["b"], ax)
            va, vb = f.loc[rows, ca].to_numpy().copy(), f.loc[rows, cb].to_numpy().copy()
            f.loc[rows, ca], f.loc[rows, cb] = vb, va
    return TrackingData(f, td.players, td.pitch, td.source + "+identity_swaps_applied", dict(td.attack_dir))
