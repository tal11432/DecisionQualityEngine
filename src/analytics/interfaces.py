"""Interfaces for FUTURE modules. Nothing here is implemented or validated.

The Decision Quality (DQ) research design:  DQ = Q(selected) - max_a Q(alternatives)
where Q is expected attacking value (e.g. EPV-based). Decision and Execution Quality
are separate quantities and must never be folded into one another:
  * DQ  depends only on the *choice* among available actions (pre-execution state);
  * EQ  depends on how the chosen action was *carried out* (e.g. realised vs. intended target);
  * outcome is recorded separately and is used to train/validate models, not to define DQ.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from src.schema import Snapshot


class Action(Protocol):
    kind: str  # 'pass' | 'shot' | 'carry' | ...


class ActionValueModel(Protocol):
    """Q(state, action): expected attacking value (xG, EPV, ...)."""
    def value(self, state: Snapshot, action: Action) -> float: ...


class AlternativeActionGenerator(Protocol):
    """Enumerates actions that were available in ``state`` (needs pass-feasibility)."""
    def alternatives(self, state: Snapshot, player_id: str) -> Sequence[Action]: ...


class PassCompletionModel(Protocol):
    def probability(self, state: Snapshot, passer: str, receiver: str) -> float: ...


class ExecutionQualityModel(Protocol):
    """How well the selected action was executed. Independent of DQ."""
    def score(self, state: Snapshot, action: Action, realised: dict) -> float: ...


def decision_quality(q_selected: float, q_alternatives: Sequence[float]) -> float:
    """DQ = Q(selected) - max(Q(alternatives)). Alternatives exclude the selected action.

    Negative => a better option existed. Requires >=1 alternative. Pure arithmetic: its
    meaning depends entirely on how Q and the alternatives were produced.
    """
    if len(q_alternatives) == 0:
        raise ValueError("Decision Quality needs at least one alternative action")
    return float(q_selected - max(q_alternatives))
