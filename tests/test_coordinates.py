import numpy as np
import pytest

from src.processing.coordinates import infer_attack_direction, metrica_to_metric
from src.schema import Pitch

P = Pitch()


def test_corners_and_centre():
    x, y = metrica_to_metric([0, 1, 0.5, 0], [0, 1, 0.5, 1], P)
    assert list(x) == [-52.5, 52.5, 0, -52.5]
    assert list(y) == [34, -34, 0, -34]  # y flipped: Metrica top (0) -> +34


def test_custom_pitch_size():
    x, y = metrica_to_metric(1.0, 0.0, Pitch(100, 60))
    assert (float(x), float(y)) == (50.0, 30.0)


def test_nan_propagates():
    x, y = metrica_to_metric(np.nan, 0.2, P)
    assert np.isnan(x) and np.isfinite(y)


def test_attack_direction_inference(metrica_files):
    from src.ingestion.metrica import read_tracking
    td = read_tracking(*metrica_files[:2], P)
    # Home starts on the left in P1 -> attacks +x; swaps in P2
    assert td.attack_dir == {("Home", 1): 1, ("Away", 1): -1, ("Home", 2): -1, ("Away", 2): 1}


def test_attack_direction_ambiguous_raises():
    import pandas as pd
    f = pd.DataFrame({"period": [1], **{f"{t}{i}": [5.0] for t in "HA" for i in range(5)}})
    cols = {t: [f"{t}{i}" for i in range(5)] for t in "HA"}  # both teams on same side
    with pytest.raises(ValueError, match="Ambiguous"):
        infer_attack_direction(f, cols, 1)


def test_snapshot_orientation(metrica_files):
    from src.ingestion.metrica import read_tracking
    td = read_tracking(*metrica_files[:2], P)
    raw = td.snapshot(0)
    away = td.snapshot(0, attacking_team="Away")  # Away attacks -x as recorded -> rotated
    gk_raw = raw.players.query("player_id == 'Player11'").iloc[0]
    gk = away.players.query("player_id == 'Player11'").iloc[0]
    assert (gk["x"], gk["y"]) == (-gk_raw["x"], -gk_raw["y"])
    home = td.snapshot(0, attacking_team="Home")  # already attacking +x -> unchanged
    assert home.players["x"].tolist() == raw.players["x"].tolist()
