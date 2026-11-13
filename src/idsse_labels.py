"""IDSSE pass eligibility (frozen methodology) and the two PROPOSED interception definitions, implemented independently.

Frozen eligibility (applied before any labelling): open-play ``Play/Pass`` only; excluded: crosses, set pieces (KickOff, ThrowIn,
GoalKick, CornerKick, FreeKick, Penalty), through balls (``Pass@Direction == throughBall``), deleted events.
NOT applicable: "exclude headers". The IDSSE pass vocabulary has no header flag (``header``/``head`` occur only on shot/assist
attributes), so headers cannot be excluded from passes; no proxy rule is invented. See reports/idsse_pilot.md.

Definition A (provider attribute): intercepted = unsuccessful pass whose ``Recipient`` belongs to the OPPOSING team.
Definition B (following event):    intercepted = unsuccessful pass followed, before any other ``Play`` event and within
                                    ``B_WINDOW_S`` seconds, by an opposing-team ``BallClaiming`` with ``Type == InterceptedBall``.
Completed (both definitions): evaluation in {successfullyCompleted, successful} and a ``Recipient`` of the SAME team.
Everything else is UNKNOWN and excluded with a reason; nothing is forced into a class.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

COMPLETED_EVAL = ("successfullyCompleted", "successful")
B_WINDOW_S = 10.0          # a-priori association window for definition B


def frozen_eligibility(ev: pd.DataFrame) -> pd.DataFrame:
    """Adds ``eligible`` and ``excluded_reason`` (first matching rule) for every Play event; non-Play events are not passes."""
    out = ev.copy()
    reason = pd.Series(None, index=out.index, dtype=object)
    reason[out["is_delete"]] = "deleted event"
    play = out["is_play"] & ~out["is_delete"]
    reason[play & out["set_piece"].notna()] = "set piece: " + out["set_piece"].astype(str)
    reason[play & (out["leaf"] == "Cross") & reason.isna()] = "cross"
    reason[play & (out["direction"] == "throughBall") & reason.isna()] = "through ball"
    out["excluded_reason"] = reason.where(play | out["is_delete"], "not a pass event")
    out["eligible"] = play & reason.isna()
    return out


def _team_of(td_players: pd.DataFrame) -> dict[str, str]:
    return dict(zip(td_players["player_id"], td_players["team"]))


def label_a(ev: pd.DataFrame, person_team: dict[str, str]) -> pd.DataFrame:
    """Definition A for every eligible pass. Columns: outcome ('completed'|'intercepted'|None) and reason."""
    rows = []
    for _, e in ev[ev["eligible"]].iterrows():
        rec_team = person_team.get(e["recipient"]) if pd.notna(e["recipient"]) else None
        if e["evaluation"] in COMPLETED_EVAL:
            if rec_team is None:
                rows.append((e["event_idx"], None, "unknown: completed pass without a (tracked) recipient"))
            elif rec_team != e["team"]:
                rows.append((e["event_idx"], None, "unknown: completed pass whose recipient is on the opposing team"))
            else:
                rows.append((e["event_idx"], "completed", ""))
        elif e["evaluation"] == "unsuccessful":
            if rec_team is None:
                rows.append((e["event_idx"], None, "unknown: unsuccessful pass without a (tracked) recipient"))
            elif rec_team != e["team"]:
                rows.append((e["event_idx"], "intercepted", ""))
            else:
                rows.append((e["event_idx"], None, "unknown: unsuccessful pass with a same-team recipient"))
        else:
            rows.append((e["event_idx"], None, f"unknown: evaluation {e['evaluation']!r}"))
    return pd.DataFrame(rows, columns=["event_idx", "outcome", "reason"])


def label_b(ev: pd.DataFrame, person_team: dict[str, str]) -> pd.DataFrame:
    """Definition B for every eligible pass (same completed rule as A; interception from a following opposing InterceptedBall claim)."""
    live = ev[~ev["is_delete"]].sort_values(["time_s", "event_idx"]).reset_index(drop=True)
    raw = [json.loads(r) for r in live["raw"]]
    rows = []
    for i, e in live.iterrows():
        if not (e["is_play"] and e["eligible"]):
            continue
        rec_team = person_team.get(e["recipient"]) if pd.notna(e["recipient"]) else None
        if e["evaluation"] in COMPLETED_EVAL:
            if rec_team is None:
                rows.append((e["event_idx"], None, "unknown: completed pass without a (tracked) recipient"))
            elif rec_team != e["team"]:
                rows.append((e["event_idx"], None, "unknown: completed pass whose recipient is on the opposing team"))
            else:
                rows.append((e["event_idx"], "completed", ""))
            continue
        if e["evaluation"] != "unsuccessful":
            rows.append((e["event_idx"], None, f"unknown: evaluation {e['evaluation']!r}"))
            continue
        found = False
        for j in range(i + 1, len(live)):
            f = live.iloc[j]
            if f["time_s"] - e["time_s"] > B_WINDOW_S or f["is_play"]:
                break
            if f["chain"].startswith("BallClaiming") and raw[j].get("BallClaiming.Type") == "InterceptedBall" and f["team"] not in (None, e["team"]):
                found = True
                break
        rows.append((e["event_idx"], "intercepted" if found else None, "" if found else "unknown: unsuccessful pass without an opposing InterceptedBall claim"))
    return pd.DataFrame(rows, columns=["event_idx", "outcome", "reason"])


def compare(la: pd.DataFrame, lb: pd.DataFrame) -> dict:
    """Agreement statistics between the two definitions over the eligible passes."""
    m = la.merge(lb, on="event_idx", suffixes=("_a", "_b"))
    a_i, b_i = m["outcome_a"] == "intercepted", m["outcome_b"] == "intercepted"
    both, only_a, only_b = int((a_i & b_i).sum()), int((a_i & ~b_i).sum()), int((~a_i & b_i).sum())
    union = both + only_a + only_b
    n = len(m)
    p_o = float(((a_i == b_i)).mean())
    p_e = float(a_i.mean() * b_i.mean() + (1 - a_i.mean()) * (1 - b_i.mean()))
    return {"n_eligible_passes": n, "A_intercepted": int(a_i.sum()), "B_intercepted": int(b_i.sum()), "both": both, "only_A": only_a, "only_B": only_b,
            "jaccard": both / union if union else float("nan"), "kappa_intercepted": (p_o - p_e) / (1 - p_e) if p_e < 1 else float("nan"),
            "A_completed": int((m["outcome_a"] == "completed").sum()), "B_completed": int((m["outcome_b"] == "completed").sum()),
            "completed_agree": int(((m["outcome_a"] == "completed") & (m["outcome_b"] == "completed")).sum()),
            "A_unknown": int(m["outcome_a"].isna().sum()), "B_unknown": int(m["outcome_b"].isna().sum())}


# material-disagreement rule, fixed BEFORE the labels were computed
JACCARD_MIN = 0.7
