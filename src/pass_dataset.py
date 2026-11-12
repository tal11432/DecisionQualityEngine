"""Leakage-safe construction of pass-decision samples.

DECISION-TIME CONTRACT
  decision frame  t_d = (event release frame) + offset_frames, offset <= 0.
  MODEL INPUTS (origin, target, opponent positions, opponent velocities) are computed from
  frames <= t_d only:
    origin     passer position at t_d
    target     position at t_d of the teammate the pass was aimed at (see below)
    opponents  positions at t_d; velocities by CAUSAL backward difference over frames t_d-w..t_d
  They never read: event end position/time/frame, interception location, receiver position at
  reception, any frame after t_d, or subsequent events.

LABELS come from event type/subtype only: completed = PASS (no subtype);
intercepted = BALL LOST with subtype exactly INTERCEPTION (that event is the failed pass).

TARGET IDENTIFICATION (label side, not a model input). Metrica gives no intended receiver for
intercepted passes (``To`` is empty) and no concept of space targets. The aimed-at teammate is
therefore identified, by one rule applied identically to every candidate, from the ball's heading
over the first ``heading_window_frames`` after release (a short window so that opponent contact is
unlikely to have bent the path). The heading selects WHICH decision-time teammate position is the
target; it is never used as a feature. Passes whose heading matches no teammate (passes into space,
unidentified targets) or matches several (ambiguous) are excluded and counted. The true receiver
(``to_player``, completed passes only) is used solely to measure that rule's accuracy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.processing.sync import synchronize_events
from src.schema import TrackingData


@dataclass(frozen=True)
class DecisionContract:
    offset_frames: int = 0
    velocity_window_frames: int = 5
    possession_max_ball_dist_m: float = 3.0
    min_pass_length_m: float = 3.0
    heading_window_frames: int = 6
    heading_min_displacement_m: float = 1.0
    heading_max_deviation_deg: float = 15.0
    heading_min_gap_deg: float = 5.0

    @classmethod
    def from_config(cls, cfg: dict) -> "DecisionContract":
        c = cfg["decision_time"]
        return cls(int(c["offset_frames"]), int(c["velocity_window_frames"]), float(c["possession_max_ball_dist_m"]),
                   float(c["min_pass_length_m"]), int(c["heading_window_frames"]),
                   float(c["heading_min_displacement_m"]), float(c["heading_max_deviation_deg"]),
                   float(c["heading_min_gap_deg"]))

    def __post_init__(self):
        if self.offset_frames > 0:
            raise ValueError("offset_frames must be <= 0 (decision cannot be after release)")
        if self.velocity_window_frames < 1 or self.heading_window_frames < 1:
            raise ValueError("windows must be >= 1 frame")


@dataclass(frozen=True)
class DecisionSample:
    event_idx: int
    label: int                    # 1 = intercepted, 0 = completed
    team: str
    period: int
    decision_row: int
    origin: tuple[float, float]
    target: tuple[float, float]   # decision-time position of the aimed-at teammate
    receiver_id: str
    length: float
    opp_xy: np.ndarray            # (m,2) tracked opponents at t_d
    opp_v: np.ndarray             # (m,2) causal velocities, NaN = unknown (treated as static)


@dataclass
class BuildResult:
    samples: list[DecisionSample]
    exclusions: pd.DataFrame      # event_idx, label, reason (first failing check)
    assignments: pd.DataFrame     # event_idx, label, assigned, true_receiver  (diagnostics only)

    def counts(self) -> pd.DataFrame:
        """Rows = reason (plus 'KEPT'), columns = completed / intercepted."""
        kept = Counter(s.label for s in self.samples)
        ex = self.exclusions.groupby(["reason", "label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        ex.loc["KEPT"] = [kept.get(0, 0), kept.get(1, 0)]
        ex.columns = ["completed", "intercepted"]
        return ex


def assign_target_by_heading(origin, mates_xy: np.ndarray, heading: np.ndarray, c: DecisionContract):
    """Teammate whose bearing from ``origin`` best matches ``heading`` (label-side rule).

    Returns (index into ``mates_xy`` or None, exclusion reason or None). Uses only the supplied
    geometry: no event end point, no outcome.
    """
    v = np.asarray(mates_xy, float) - np.asarray(origin, float)
    ang = np.degrees(np.abs(np.arctan2(heading[0] * v[:, 1] - heading[1] * v[:, 0], v @ heading)))
    order = np.argsort(ang)
    best = ang[order[0]]
    gap = ang[order[1]] - best if len(order) > 1 else np.inf
    if best > c.heading_max_deviation_deg:
        return None, "no teammate within heading cone (pass into space / unidentified target)"
    if gap < c.heading_min_gap_deg:
        return None, "ambiguous receiver (runner-up within heading gap)"
    return int(order[0]), None


def candidate_events(events: pd.DataFrame) -> pd.DataFrame:
    """In-universe events with label from type/subtype ONLY. Other columns are not read here."""
    comp = events[(events["type"] == "PASS") & events["subtype"].isna()].assign(label=0)
    icpt = events[(events["type"] == "BALL LOST") & (events["subtype"] == "INTERCEPTION")].assign(label=1)
    return pd.concat([comp, icpt]).sort_index()


def build_decision_samples(td: TrackingData, events: pd.DataFrame, tolerance_s: float,
                           contract: DecisionContract = DecisionContract()) -> BuildResult:
    sel = candidate_events(events)
    sync = synchronize_events(sel, td.frames, tolerance_s, "start_time_s", frame_col=None)
    rel_row = sync.matched.set_index("event_idx")["frame_row"].to_dict()
    sync_fail = dict(zip(sync.unmatched["event_idx"], sync.unmatched["reason"]))

    f = td.frames
    n = len(f)
    keys = list(td.players[["team", "player_id"]].itertuples(index=False, name=None))
    X = np.column_stack([f[td.col(t, p, "x")].to_numpy(float) for t, p in keys])
    Y = np.column_stack([f[td.col(t, p, "y")].to_numpy(float) for t, p in keys])
    team_of = np.array([t for t, _ in keys])
    pid_of = np.array([p for _, p in keys])
    bx, by = f["ball_x"].to_numpy(float), f["ball_y"].to_numpy(float)
    frame = f["frame"].to_numpy(dtype=float, na_value=np.nan)
    period = f["period"].to_numpy(dtype=float, na_value=np.nan)
    c = contract
    has_true = "to_player" in events.columns

    def consecutive(i0: int, i1: int, k: int) -> bool:
        return 0 <= i0 < n and 0 <= i1 < n and period[i0] == period[i1] and frame[i1] - frame[i0] == k

    samples: list[DecisionSample] = []
    excl: list[tuple[int, int, str]] = []
    assign: list[tuple[int, int, str | None, str | None]] = []

    for idx, e in sel.iterrows():
        lab, team = int(e["label"]), e["team"]

        def skip(reason: str):
            excl.append((idx, lab, reason))

        if idx not in rel_row:
            skip("no tracking frame at event time: " + sync_fail[idx].split(" (")[0])
            continue
        r_rel = rel_row[idx]
        r_d = r_rel + c.offset_frames
        if not consecutive(r_d, r_rel, -c.offset_frames):
            skip("decision frame unavailable (data boundary/gap)")
            continue
        j = [k for k, (t, p) in enumerate(keys) if t == team and p == e["from_player"]]
        if not j or not np.isfinite([X[r_d, j[0]], Y[r_d, j[0]]]).all():
            skip("passer not tracked at decision frame")
            continue
        j = j[0]
        origin = np.array([X[r_d, j], Y[r_d, j]])
        if not np.isfinite([bx[r_d], by[r_d]]).all():
            skip("ball not tracked at decision frame")
            continue
        if np.hypot(bx[r_d] - origin[0], by[r_d] - origin[1]) > c.possession_max_ball_dist_m:
            skip("ball not with passer at decision frame")
            continue
        # velocity history (causal)
        w = c.velocity_window_frames
        if not consecutive(r_d - w, r_d, w):
            skip("velocity history unavailable")
            continue
        # ---- label-side target identification from post-release ball heading ----
        r_h = r_rel + c.heading_window_frames
        if not (consecutive(r_rel, r_h, c.heading_window_frames)
                and np.isfinite([bx[r_rel], by[r_rel], bx[r_h], by[r_h]]).all()):
            skip("post-release ball track unavailable")
            continue
        h = np.array([bx[r_h] - bx[r_rel], by[r_h] - by[r_rel]])
        if np.hypot(*h) < c.heading_min_displacement_m:
            skip("ball displacement after release too small for heading")
            continue
        mates = np.flatnonzero((team_of == team) & (np.arange(len(keys)) != j)
                               & np.isfinite(X[r_d]) & np.isfinite(Y[r_d]))
        if len(mates) == 0:
            skip("no teammate tracked at decision frame")
            continue
        pick, why = assign_target_by_heading(origin, np.column_stack([X[r_d, mates], Y[r_d, mates]]), h, c)
        true = e["to_player"] if has_true and pd.notna(e["to_player"]) else None
        if pick is None:
            assign.append((idx, lab, None, true))
            skip(why)
            continue
        k = mates[pick]
        assign.append((idx, lab, pid_of[k], true))
        target = np.array([X[r_d, k], Y[r_d, k]])
        length = float(np.hypot(*(target - origin)))
        if length < c.min_pass_length_m:
            skip("target closer than minimum pass length")
            continue
        # ---- decision-time inputs ----
        opp = np.flatnonzero((team_of != team) & np.isfinite(X[r_d]) & np.isfinite(Y[r_d]))
        if len(opp) == 0:
            skip("no opponents tracked at decision frame")
            continue
        t0, t1 = f["time_s"].iloc[r_d - w], f["time_s"].iloc[r_d]
        dt = float(t1 - t0)
        ovx = (X[r_d, opp] - X[r_d - w, opp]) / dt
        ovy = (Y[r_d, opp] - Y[r_d - w, opp]) / dt
        opp_v = np.column_stack([ovx, ovy])
        if not np.isfinite(opp_v).all(axis=1).any():
            skip("no opponent velocity available")
            continue
        samples.append(DecisionSample(idx, lab, team, int(period[r_d]), r_d, (float(origin[0]), float(origin[1])),
                                      (float(target[0]), float(target[1])), str(pid_of[k]), length,
                                      np.column_stack([X[r_d, opp], Y[r_d, opp]]), opp_v))
    return BuildResult(samples,
                       pd.DataFrame(excl, columns=["event_idx", "label", "reason"]),
                       pd.DataFrame(assign, columns=["event_idx", "label", "assigned", "true_receiver"]))
