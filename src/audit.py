"""Leak demonstration ONLY. Never import from modelling code.

Re-scores the SAME decisions with the target replaced by the event's actual end point (the
interception location for intercepted passes), i.e. the discarded legacy protocol, to quantify how
much that outcome-derived target inflates a score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.feasibility import FeasibilityParams, interception_margin
from src.pass_dataset import DecisionSample


def endpoint_target_margins(samples: list[DecisionSample], events: pd.DataFrame, params: FeasibilityParams,
                            use_velocity: bool = False) -> np.ndarray:
    """LEAKY: margin to the actual event end point. Samples with a missing end point get NaN."""
    out = []
    for s in samples:
        e = events.loc[s.event_idx]
        if not np.isfinite([e["end_x"], e["end_y"]]).all():
            out.append(np.nan)
            continue
        out.append(interception_margin(s.origin, (float(e["end_x"]), float(e["end_y"])), s.opp_xy, params,
                                       s.opp_v if use_velocity else None)["margin_s"])
    return np.array(out, float)
