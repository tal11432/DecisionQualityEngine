"""Labels and decision/prediction-time features for pass completion. Two SEPARATE scenarios.

SCENARIO A  "decision" (target-free).  Prediction frame t_d = ball-release frame.
  Features use frames <= t_d only. The intended receiver is NOT knowable at or before release
  (Metrica gives no receiver for failed passes; no body orientation; see ``target_identifiability``),
  so A uses passer-state, local pressure and option-set aggregates over ALL teammates. No
  target-identification step exists, hence no target-related exclusions or selection.

SCENARIO B  "early flight" (identified target).  Prediction frame t_p = t_release + K frames (K=6, 0.24 s).
  Frames <= t_p are used. The intended receiver is identified from the ball's heading over the first K frames
  (``pass_dataset.assign_target_by_heading``). This is NOT a decision-time prediction: the ball has been in
  flight for 0.24 s. Passes with unidentified / ambiguous targets (incl. passes into space) are excluded.

Outcome labels (event type/subtype only; applied identically in both scenarios):
  completed   = PASS, no subtype, receiver present, positive duration
  intercepted = BALL LOST, subtype exactly INTERCEPTION, passer present (that event IS the failed pass)
  anything else pass-related (PASS with subtype, other BALL LOST subtypes incl. THEFT/none/HEAD...)
  is UNKNOWN: excluded and counted, never relabelled as an interception. Labels are NOT conditioned on following
  events (doing so would e.g. drop completed passes whose receiver was immediately challenged).
Features never read: end position/time/frame, interception location, receiver identity (``to_player``), later events.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.analytics.geometry import angle_to_goal, passing_lane_obstruction
from src.pass_dataset import DecisionContract, assign_target_by_heading
from src.processing.sync import synchronize_events
from src.schema import TrackingData

# Columns the feature builders are allowed to see. Everything else in an events table is outcome info.
ALLOWED_EVENT_COLUMNS = ["team", "period", "start_time_s", "from_player"]
LANE_WIDTH_M = 3.0
MARGIN_CLIP_S = 10.0

# Registry used for documentation and leakage tests: name -> (scenario, unit, definition, timestamp)
FEATURES: dict[str, tuple[str, str, str, str]] = {
    # --- scenario A (decision time, frames <= t_d) ---
    "dist_goal": ("A", "m", "passer distance to centre of the goal being attacked", "t_d"),
    "angle_goal": ("A", "rad", "angle subtended by the posts at the passer (shooting aperture)", "t_d"),
    "nearest_opp_dist": ("A", "m", "distance from passer to nearest tracked opponent", "t_d"),
    "n_opp_10m": ("A", "count", "tracked opponents within 10 m of the passer", "t_d"),
    "passer_speed": ("A", "m/s", "passer speed, causal 5-frame backward difference", "t_d-5..t_d"),
    "nearest_opp_closing_speed": ("A", "m/s", "rate at which the nearest opponent approaches the passer (+ = closing), causal", "t_d-5..t_d"),
    "n_safe_options": ("A", "count", "teammates (>=3 m away) whose lane from the passer is NOT interceptable (margin<0)", "t_d"),
    "best_margin": ("A", "s", "most negative interception margin over all teammate lanes (safest option); clipped +-10", "t_d"),
    # --- scenario B (early flight, frames <= t_p = release + 6) ---
    "pass_dist": ("B", "m", "distance from ball release point to the target teammate's position at t_p", "t_rel, t_p"),
    "early_ball_speed": ("B", "m/s", "ball displacement over the first 6 frames / 0.24 s", "t_rel..t_p"),
    "margin_static": ("B", "s", "interception margin, ball(t_p)->target(t_p), static opponents, prior ball speed", "t_p"),
    "margin_vel": ("B", "s", "as margin_static but opponents advance at causal velocity during reaction time", "t_p-5..t_p"),
    "n_opp_corridor": ("B", "count", "opponents within 1.5 m of the remaining pass line ball(t_p)->target(t_p)", "t_p"),
    "min_opp_corridor_dist": ("B", "m", "smallest opponent distance to that line segment", "t_p"),
    "receiver_speed": ("B", "m/s", "target teammate speed, causal 5-frame backward difference (history only)", "t_p-5..t_p"),
    "receiver_nearest_opp_dist": ("B", "m", "distance from target teammate to nearest opponent", "t_p"),
    "forward_gain": ("B", "m", "passer distance-to-goal at release minus target distance-to-goal at t_p", "t_rel, t_p"),
    "dist_goal_target": ("B", "m", "target teammate distance to the goal being attacked", "t_p"),
}
FEATURES_A = [k for k, v in FEATURES.items() if v[0] == "A"]
FEATURES_B = [k for k, v in FEATURES.items() if v[0] == "B"]


# ---------------------------------------------------------------- labels
def label_events(events: pd.DataFrame) -> pd.DataFrame:
    """Outcome per pass-related event. Columns: event_idx, outcome ('completed'|'intercepted'|None), reason.

    Only PASS and BALL LOST events are pass-related; SET PIECE / RECOVERY / CHALLENGE / SHOT ... are not
    classified at all (not unknown, simply out of scope).
    """
    rows = []
    for idx, e in events[events["type"].isin(["PASS", "BALL LOST"])].iterrows():
        sub = e["subtype"] if pd.notna(e["subtype"]) else None
        if e["type"] == "PASS":
            if sub is not None:
                rows.append((idx, None, f"unknown: PASS with subtype ({'aerial/special' if sub else ''})"))
            elif pd.isna(e["to_player"]):
                rows.append((idx, None, "unknown: PASS without receiver"))
            elif not e["end_time_s"] > e["start_time_s"]:
                rows.append((idx, None, "unknown: PASS with non-positive duration"))
            else:
                rows.append((idx, "completed", ""))
        else:
            if sub == "INTERCEPTION":
                if pd.isna(e["from_player"]):
                    rows.append((idx, None, "unknown: INTERCEPTION without passer"))
                else:
                    rows.append((idx, "intercepted", ""))
            elif sub is None:
                rows.append((idx, None, "unknown: possession loss without subtype (not an identified pass interception)"))
            elif "INTERCEPTION" in sub:
                rows.append((idx, None, "unknown: non-ground interception (cross/head/goal kick)"))
            else:
                rows.append((idx, None, f"unknown: possession loss '{sub}' (not a pass interception)"))
    return pd.DataFrame(rows, columns=["event_idx", "outcome", "reason"])


# ---------------------------------------------------------------- tracking arrays
@dataclass
class _Track:
    X: np.ndarray
    Y: np.ndarray
    team: np.ndarray
    pid: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    frame: np.ndarray
    period: np.ndarray
    time: np.ndarray
    attack: dict
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.frame)

    @classmethod
    def of(cls, td: TrackingData) -> "_Track":
        f = td.frames
        keys = list(td.players[["team", "player_id"]].itertuples(index=False, name=None))
        return cls(np.column_stack([f[td.col(t, p, "x")].to_numpy(float) for t, p in keys]),
                   np.column_stack([f[td.col(t, p, "y")].to_numpy(float) for t, p in keys]),
                   np.array([t for t, _ in keys]), np.array([p for _, p in keys]),
                   f["ball_x"].to_numpy(float), f["ball_y"].to_numpy(float),
                   f["frame"].to_numpy(dtype=float, na_value=np.nan), f["period"].to_numpy(dtype=float, na_value=np.nan),
                   f["time_s"].to_numpy(float), td.attack_dir)

    def consecutive(self, i0: int, i1: int, k: int) -> bool:
        return (0 <= i0 < self.n and 0 <= i1 < self.n and self.period[i0] == self.period[i1]
                and self.frame[i1] - self.frame[i0] == k)

    def sign(self, team: str, row: int) -> int:
        return self.attack[(team, int(self.period[row]))]


@dataclass
class FeatureSet:
    table: pd.DataFrame        # event_idx, y_completed, team, period, + feature columns (+ post-hoc stratifier cols prefixed 'ph_')
    exclusions: pd.DataFrame   # event_idx, outcome, reason
    unknown: pd.DataFrame      # event_idx, reason  (outcome unknown; not candidates)
    scenario: str


def _goal(pitch_len: float) -> tuple[float, float]:
    return (pitch_len / 2, 0.0)


def build_features(td: TrackingData, events: pd.DataFrame, scenario: str, tolerance_s: float,
                   contract: DecisionContract, params: FeasibilityParams) -> FeatureSet:
    """Build the feature table for ``scenario`` 'A' or 'B'. ``events`` is the full table; only
    ALLOWED_EVENT_COLUMNS are passed to the feature code, labels come from ``label_events``."""
    if scenario not in ("A", "B"):
        raise ValueError("scenario must be 'A' or 'B'")
    labels = label_events(events)
    unknown = labels[labels["outcome"].isna()][["event_idx", "reason"]].reset_index(drop=True)
    cand = labels[labels["outcome"].notna()].set_index("event_idx")["outcome"]
    ev = events.loc[cand.index, ALLOWED_EVENT_COLUMNS]          # <-- the ONLY event data features can see
    sync = synchronize_events(ev, td.frames, tolerance_s, "start_time_s", frame_col=None)
    rel = sync.matched.set_index("event_idx")["frame_row"].to_dict()
    why_sync = dict(zip(sync.unmatched["event_idx"], sync.unmatched["reason"]))
    T = _Track.of(td)
    c, w, K = contract, contract.velocity_window_frames, contract.heading_window_frames
    L = td.pitch.length
    goal = _goal(L)
    rows, excl = [], []

    for idx, e in ev.iterrows():
        outcome = cand[idx]

        def skip(reason):
            excl.append((idx, outcome, reason))

        if idx not in rel:
            skip("no tracking frame at event time: " + why_sync[idx].split(" (")[0])
            continue
        r0 = rel[idx]
        r = r0 if scenario == "A" else r0 + K          # prediction frame
        team = e["team"]
        if not T.consecutive(r0, r, r - r0) or not T.consecutive(r - w, r, w):
            skip("prediction frame or velocity history unavailable")
            continue
        j = [k for k in range(len(T.pid)) if T.team[k] == team and T.pid[k] == e["from_player"]]
        if not j or not np.isfinite([T.X[r0, j[0]], T.Y[r0, j[0]]]).all():
            skip("passer not tracked at release")
            continue
        j = j[0]
        if not np.isfinite([T.bx[r0], T.by[r0]]).all():
            skip("ball not tracked at release")
            continue
        origin = np.array([T.X[r0, j], T.Y[r0, j]])
        if np.hypot(T.bx[r0] - origin[0], T.by[r0] - origin[1]) > c.possession_max_ball_dist_m:
            skip("ball not with passer at release")
            continue
        sgn = T.sign(team, r0)                          # orientation: +x = attacking direction
        dt = float(T.time[r] - T.time[r - w])
        ok_now = np.isfinite(T.X[r]) & np.isfinite(T.Y[r])
        opp = np.flatnonzero((T.team != team) & ok_now)
        mates = np.flatnonzero((T.team == team) & ok_now & (np.arange(len(T.pid)) != j))
        if len(opp) == 0:
            skip("no opponents tracked at prediction frame")
            continue
        opp_xy = np.column_stack([T.X[r, opp], T.Y[r, opp]])
        opp_v = np.column_stack([T.X[r, opp] - T.X[r - w, opp], T.Y[r, opp] - T.Y[r - w, opp]]) / dt

        def to_att(xy):  # rotate into attacking orientation
            return np.asarray(xy, float) * sgn

        feat: dict[str, float] = {}
        if scenario == "A":
            if not np.isfinite([T.X[r - w, j], T.Y[r - w, j]]).all():
                skip("passer velocity unavailable")
                continue
            p_att = to_att(origin)
            d = np.hypot(*(opp_xy - origin).T)
            n = int(np.argmin(d))
            prev = np.hypot(T.X[r - w, opp[n]] - T.X[r - w, j], T.Y[r - w, opp[n]] - T.Y[r - w, j])
            if not np.isfinite(prev):
                skip("nearest opponent history unavailable")
                continue
            elig = [k for k in mates if np.hypot(T.X[r, k] - origin[0], T.Y[r, k] - origin[1]) >= c.min_pass_length_m]
            if not elig:
                skip("no eligible teammate options")
                continue
            margins = np.array([interception_margin(origin, (T.X[r, k], T.Y[r, k]), opp_xy, params, opp_v)["margin_s"]
                                for k in elig])
            margins = np.clip(margins, -MARGIN_CLIP_S, MARGIN_CLIP_S)
            feat = {
                "dist_goal": float(np.hypot(*(np.array(goal) - p_att))),
                "angle_goal": float(angle_to_goal(p_att, goal, td.pitch.goal_width)),
                "nearest_opp_dist": float(d.min()),
                "n_opp_10m": int((d <= 10.0).sum()),
                "passer_speed": float(np.hypot(T.X[r, j] - T.X[r - w, j], T.Y[r, j] - T.Y[r - w, j]) / dt),
                "nearest_opp_closing_speed": float((prev - d[n]) / dt),
                "n_safe_options": int((margins < 0).sum()),
                "best_margin": float(margins.min()),
            }
        else:
            if not (T.consecutive(r0, r, K) and np.isfinite([T.bx[r], T.by[r]]).all()):
                skip("post-release ball track unavailable")
                continue
            h = np.array([T.bx[r] - T.bx[r0], T.by[r] - T.by[r0]])
            if np.hypot(*h) < c.heading_min_displacement_m:
                skip("ball displacement after release too small for heading")
                continue
            m0 = np.flatnonzero((T.team == team) & (np.arange(len(T.pid)) != j)
                                & np.isfinite(T.X[r0]) & np.isfinite(T.Y[r0]))
            if len(m0) == 0:
                skip("no teammate tracked at release")
                continue
            pick, why = assign_target_by_heading(origin, np.column_stack([T.X[r0, m0], T.Y[r0, m0]]), h, c)
            if pick is None:
                skip(why)
                continue
            k = m0[pick]
            if not np.isfinite([T.X[r, k], T.Y[r, k], T.X[r - w, k], T.Y[r - w, k]]).all():
                skip("target not tracked (with history) at prediction frame")
                continue
            tgt = np.array([T.X[r, k], T.Y[r, k]])
            rel_pt = np.array([T.bx[r0], T.by[r0]])
            pass_dist = float(np.hypot(*(tgt - rel_pt)))
            if pass_dist < c.min_pass_length_m:
                skip("target closer than minimum pass length")
                continue
            ball_now = np.array([T.bx[r], T.by[r]])
            lane = passing_lane_obstruction(tuple(ball_now), tuple(tgt), opp_xy, lane_width=2 * 1.5)
            rv = (tgt - np.array([T.X[r - w, k], T.Y[r - w, k]])) / dt
            m_static = interception_margin(ball_now, tgt, opp_xy, params)["margin_s"]
            m_vel = interception_margin(ball_now, tgt, opp_xy, params, opp_v)["margin_s"]
            feat = {
                "pass_dist": pass_dist,
                "early_ball_speed": float(np.hypot(*h) / (K * float(np.median(np.diff(T.time[r0:r + 1]))))),
                "margin_static": float(np.clip(m_static, -MARGIN_CLIP_S, MARGIN_CLIP_S)),
                "margin_vel": float(np.clip(m_vel, -MARGIN_CLIP_S, MARGIN_CLIP_S)),
                "n_opp_corridor": int(lane["n_blockers"]),
                "min_opp_corridor_dist": float(lane["min_distance"]),
                "receiver_speed": float(np.hypot(*rv)),
                "receiver_nearest_opp_dist": float(np.hypot(*(opp_xy - tgt).T).min()),
                "forward_gain": float(np.hypot(*(np.array(goal) - to_att(origin))) - np.hypot(*(np.array(goal) - to_att(tgt)))),
                "dist_goal_target": float(np.hypot(*(np.array(goal) - to_att(tgt)))),
            }
        rows.append({"event_idx": idx, "y_completed": int(outcome == "completed"), "team": team,
                     "period": int(T.period[r]), "row": r, **feat})
    cols = ["event_idx", "y_completed", "team", "period", "row"] + (FEATURES_A if scenario == "A" else FEATURES_B)
    return FeatureSet(pd.DataFrame(rows, columns=cols), pd.DataFrame(excl, columns=["event_idx", "outcome", "reason"]),
                      unknown, scenario)


def exclusion_counts(fs: FeatureSet) -> pd.DataFrame:
    """Rows = reason (+KEPT, +UNKNOWN OUTCOME reasons), columns = completed / intercepted (unknown reported separately)."""
    ex = fs.exclusions.groupby(["reason", "outcome"]).size().unstack(fill_value=0).reindex(
        columns=["completed", "intercepted"], fill_value=0)
    ex.loc["KEPT"] = [int(fs.table["y_completed"].sum()), int((1 - fs.table["y_completed"]).sum())]
    return ex


# ---------------------------------------------------------------- decision-time receiver identifiability
def target_identifiability(td: TrackingData, events: pd.DataFrame, tolerance_s: float, contract: DecisionContract,
                           params: FeasibilityParams) -> dict:
    """Can the receiver be identified at/before release? Tests three fixed, a-priori rules on completed passes
    (the only passes with a known receiver, taken from ``to_player`` as ground truth for SCORING only).
    Rules use frames <= release: nearest teammate; teammate best aligned with the passer's causal velocity;
    safest lane (lowest interception margin)."""
    labels = label_events(events)
    comp = labels[labels["outcome"] == "completed"]["event_idx"]
    ev = events.loc[comp, ALLOWED_EVENT_COLUMNS + ["to_player"]]
    sync = synchronize_events(ev[ALLOWED_EVENT_COLUMNS], td.frames, tolerance_s, "start_time_s", frame_col=None)
    rel = sync.matched.set_index("event_idx")["frame_row"].to_dict()
    T = _Track.of(td)
    w = contract.velocity_window_frames
    hits = {"nearest": 0, "velocity_aligned": 0, "safest_lane": 0}
    chance, n, n_vel = 0.0, 0, 0
    for idx, e in ev.iterrows():
        if idx not in rel:
            continue
        r = rel[idx]
        if not T.consecutive(r - w, r, w):
            continue
        j = [k for k in range(len(T.pid)) if T.team[k] == e["team"] and T.pid[k] == e["from_player"]]
        ok = np.isfinite(T.X[r]) & np.isfinite(T.Y[r])
        if not j or not ok[j[0]]:
            continue
        j = j[0]
        mates = np.flatnonzero((T.team == e["team"]) & ok & (np.arange(len(T.pid)) != j))
        true = np.flatnonzero(T.pid[mates] == e["to_player"])
        if len(mates) < 2 or len(true) == 0:
            continue
        true = true[0]
        origin = np.array([T.X[r, j], T.Y[r, j]])
        vec = np.column_stack([T.X[r, mates], T.Y[r, mates]]) - origin
        n += 1
        chance += 1 / len(mates)
        hits["nearest"] += int(np.argmin(np.hypot(*vec.T)) == true)
        v = origin - np.array([T.X[r - w, j], T.Y[r - w, j]])
        if np.isfinite(v).all() and np.hypot(*v) / (w * 0.04) >= 1.0:
            n_vel += 1
            cos = (vec @ v) / (np.hypot(*vec.T) * np.hypot(*v) + 1e-9)
            hits["velocity_aligned"] += int(np.argmax(cos) == true)
        opp = np.flatnonzero((T.team != e["team"]) & ok)
        if len(opp):
            oxy = np.column_stack([T.X[r, opp], T.Y[r, opp]])
            m = [interception_margin(origin, origin + vec[i], oxy, params)["margin_s"] for i in range(len(mates))]
            hits["safest_lane"] += int(np.argmin(m) == true)
    return {"n_completed_with_receiver": n, "n_moving_passer": n_vel, "chance_top1": chance / max(n, 1),
            "top1_nearest": hits["nearest"] / max(n, 1), "top1_safest_lane": hits["safest_lane"] / max(n, 1),
            "top1_velocity_aligned_(moving passers)": hits["velocity_aligned"] / max(n_vel, 1)}
