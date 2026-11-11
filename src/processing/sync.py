"""Event <-> tracking-frame synchronization by timestamp (never by guessing)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyncResult:
    matched: pd.DataFrame    # event_idx, frame_row, frame, dt_s, frame_mismatch
    unmatched: pd.DataFrame  # event_idx, reason

    def summary(self) -> str:
        n = len(self.matched) + len(self.unmatched)
        lines = [f"Event sync: {len(self.matched)}/{n} matched, {len(self.unmatched)} unmatched"]
        if len(self.unmatched):
            for reason, c in self.unmatched["reason"].value_counts().items():
                lines.append(f"  unmatched: {c} x {reason}")
        if len(self.matched):
            lines.append(f"  max |dt| = {self.matched['dt_s'].abs().max():.4f}s; "
                         f"frame-number mismatches vs provider: {int(self.matched['frame_mismatch'].sum())}")
        return "\n".join(lines)


def synchronize_events(events: pd.DataFrame, frames: pd.DataFrame, tolerance_s: float = 0.02,
                       time_col: str = "start_time_s", frame_col: str | None = "start_frame") -> SyncResult:
    """Match each event to the nearest tracking frame within the same period.

    An event is left unmatched (with a reason) when: its time/period is missing, the period
    has no tracking, no frame lies within ``tolerance_s``, or the nearest timestamp is shared
    by several tracking rows (ambiguous). ``frame_mismatch`` flags matches whose tracking frame
    number differs from the provider's own event frame number (``frame_col``).
    Event and frame times must share a clock (seconds from period-agnostic match start, as in Metrica).
    """
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be >= 0")
    matched, unmatched = [], []
    by_period = {}
    for p, g in frames.dropna(subset=["period", "time_s"]).groupby("period"):
        g = g.sort_values("time_s", kind="stable")
        by_period[int(p)] = (g["time_s"].to_numpy(float), g.index.to_numpy(), g["frame"].to_numpy())
    pos = {idx: i for i, idx in enumerate(frames.index)}

    for idx, ev in events.iterrows():
        t, p = ev.get(time_col), ev.get("period")
        if pd.isna(t) or pd.isna(p):
            unmatched.append((idx, "missing event time or period"))
            continue
        if int(p) not in by_period:
            unmatched.append((idx, f"no tracking for period {int(p)}"))
            continue
        times, labels, fnums = by_period[int(p)]
        j = int(np.searchsorted(times, t))
        cand = [k for k in (j - 1, j) if 0 <= k < len(times)]
        k = min(cand, key=lambda c: abs(times[c] - t))
        dt = float(times[k] - t)
        if abs(dt) > tolerance_s + 1e-9:
            unmatched.append((idx, f"no frame within {tolerance_s}s (nearest dt={dt:+.3f}s)"))
            continue
        if np.count_nonzero(times == times[k]) > 1:
            unmatched.append((idx, "ambiguous: duplicate tracking timestamp"))
            continue
        pf = ev.get(frame_col) if frame_col else np.nan
        mismatch = bool(pd.notna(pf) and int(pf) != int(fnums[k]))
        matched.append((idx, pos[labels[k]], int(fnums[k]), dt, mismatch))

    return SyncResult(
        pd.DataFrame(matched, columns=["event_idx", "frame_row", "frame", "dt_s", "frame_mismatch"]),
        pd.DataFrame(unmatched, columns=["event_idx", "reason"]),
    )
