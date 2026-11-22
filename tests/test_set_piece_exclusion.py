"""Set-piece deliveries are not open-play ground passes (reports/set_piece_delivery_check.md).

Metrica records a restart as a zero-duration ``SET PIECE`` marker followed, at the same instant and by the same player,
by the ``PASS`` (or the failed-pass ``BALL LOST``) that delivers it. ``label_events`` and ``candidate_events`` must treat
that PASS / BALL LOST as the restart itself and leave it out of the open-play population.

These tests FAIL on the current code (which labels such events completed / intercepted) and pass once the proposed
change in ``reports/set_piece_delivery_exclusion.patch`` is applied. The negative controls pass both before and after.
"""
import pandas as pd

from src.pass_completion_features import label_events
from src.pass_dataset import candidate_events

COLS = ["team", "type", "subtype", "period", "start_time_s", "end_time_s", "from_player", "to_player"]


def _ev(rows):
    return pd.DataFrame(rows, columns=COLS)


def _table():
    """index: what it is."""
    return _ev([
        ("Home", "SET PIECE", "THROW IN", 1, 10.0, 10.0, "H1", None),          # 0 marker
        ("Home", "PASS", None, 1, 10.0, 11.2, "H1", "H2"),                      # 1 THE delivery        -> excluded
        ("Home", "PASS", None, 1, 12.0, 13.0, "H2", "H3"),                      # 2 open play, next event -> kept
        ("Away", "SET PIECE", "FREE KICK", 1, 20.0, 20.0, "A1", None),          # 3 marker
        ("Away", "BALL LOST", "INTERCEPTION", 1, 20.0, 21.0, "A1", None),       # 4 failed delivery     -> excluded
        ("Home", "SET PIECE", "CORNER KICK", 1, 30.0, 30.0, "H1", None),        # 5 marker
        ("Home", "PASS", None, 1, 30.6, 31.5, "H1", "H4"),                      # 6 delivery after 0.6 s -> excluded
        ("Home", "SET PIECE", "FREE KICK", 1, 40.0, 40.0, "H1", None),          # 7 marker
        ("Home", "PASS", None, 1, 41.5, 42.5, "H1", "H2"),                      # 8 1.5 s later: outside window -> kept
        ("Home", "SET PIECE", "THROW IN", 1, 50.0, 50.0, "H1", None),           # 9 marker
        ("Home", "PASS", None, 1, 50.0, 51.0, "H5", "H2"),                      # 10 different player -> kept
        ("Home", "SET PIECE", "THROW IN", 1, 60.0, 60.0, "H1", None),           # 11 marker
        ("Away", "PASS", None, 1, 60.0, 61.0, "H1", "A2"),                      # 12 different team -> kept
        ("Home", "SET PIECE", "KICK OFF", 2, 70.0, 70.0, "H1", None),           # 13 marker (period 2)
        ("Home", "PASS", None, 1, 70.0, 71.0, "H1", "H2"),                      # 14 other period -> kept
    ])


EXCLUDED = {1, 4, 6}
KEPT = {2, 8, 10, 12, 14}


def test_label_events_leaves_set_piece_deliveries_out():
    lab = label_events(_table()).set_index("event_idx")
    assert lab.loc[sorted(EXCLUDED), "outcome"].isna().all()
    assert all("set-piece delivery" in lab.loc[i, "reason"] for i in EXCLUDED)


def test_candidate_events_leaves_set_piece_deliveries_out():
    assert EXCLUDED.isdisjoint(candidate_events(_table()).index)


def test_open_play_passes_and_near_misses_are_kept():
    """Negative controls (pass before and after the change): different player / team / period or a gap > 1 s is NOT a delivery."""
    lab = label_events(_table()).set_index("event_idx")
    assert lab.loc[sorted(KEPT), "outcome"].tolist() == ["completed"] * len(KEPT)
    assert KEPT <= set(candidate_events(_table()).index)


def test_legacy_population_can_be_reproduced_explicitly():
    ev = _table()
    lab = label_events(ev, exclude_set_piece_deliveries=False).set_index("event_idx")
    assert lab.loc[1, "outcome"] == "completed" and lab.loc[4, "outcome"] == "intercepted"
    assert EXCLUDED <= set(candidate_events(ev, exclude_set_piece_deliveries=False).index)
