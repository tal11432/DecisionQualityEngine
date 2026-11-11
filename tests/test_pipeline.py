import numpy as np
import pytest

from src.ingestion.metrica import read_events, read_tracking
from src.processing.sync import synchronize_events
from src.processing.validation import FAIL, OK, WARN, validate_tracking
from src.schema import Pitch


@pytest.fixture
def td(metrica_files):
    return read_tracking(*metrica_files[:2], Pitch())


def test_loading(td, metrica_files):
    assert len(td.frames) == 5 and len(td.players) == 10
    assert td.frames["period"].tolist() == [1, 1, 1, 2, 2]
    assert td.frames[td.col("Home", "Player1", "x")].iloc[0] == pytest.approx(-47.25)  # (0.05-0.5)*105
    assert td.frames["ball_x"].iloc[0] == 0 and td.frames["ball_y"].iloc[0] == 0


def test_missing_player_and_ball_are_nan_not_zero(td):
    assert np.isnan(td.frames[td.col("Home", "Player1", "x")].iloc[1])
    assert np.isnan(td.frames["ball_x"].iloc[2])
    s = td.snapshot(1)
    assert "Player1" not in s.players["player_id"].tolist() and len(s.players) == 9
    assert np.isnan(td.snapshot(2).ball).all()


def test_missing_file_and_bad_format(tmp_path, metrica_files):
    with pytest.raises(FileNotFoundError):
        read_tracking(tmp_path / "nope.csv", metrica_files[1])
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n3,4\n5,6\n")
    with pytest.raises(ValueError, match="Metrica"):
        read_tracking(bad, metrica_files[1])
    with pytest.raises(ValueError, match="team labels"):  # Away file passed as Home
        read_tracking(metrica_files[1], metrica_files[1])


def test_misaligned_files_rejected(tmp_path, metrica_files):
    lines = metrica_files[1].read_text().splitlines()
    short = tmp_path / "short.csv"
    short.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="not aligned"):
        read_tracking(metrica_files[0], short)


def test_events_loading(metrica_files):
    ev = read_events(metrica_files[2], Pitch())
    assert len(ev) == 6 and ev.loc[0, "start_x"] == pytest.approx(-31.5)
    assert ev.loc[0, "start_y"] == pytest.approx(13.6)
    with pytest.raises(FileNotFoundError):
        read_events("missing.csv")


def test_sync(td, metrica_files):
    ev = read_events(metrica_files[2])
    res = synchronize_events(ev, td.frames, tolerance_s=0.01)
    assert res.matched["event_idx"].tolist() == [0, 1, 5]
    assert res.matched["frame"].tolist() == [2, 2, 4]
    assert res.matched.loc[1, "dt_s"] == pytest.approx(-0.001)
    reasons = dict(zip(res.unmatched["event_idx"], res.unmatched["reason"]))
    assert "no frame within" in reasons[2]
    assert "no tracking for period 3" in reasons[3]
    assert "missing event time" in reasons[4]
    assert not res.matched["frame_mismatch"].any()


def test_sync_tolerance_is_enforced(td, metrica_files):
    ev = read_events(metrica_files[2]).iloc[[1]].copy()
    ev["start_time_s"] = 0.10  # 0.02 from frames 2 and 3
    assert len(synchronize_events(ev, td.frames, tolerance_s=0.01).matched) == 0
    assert len(synchronize_events(ev, td.frames, tolerance_s=0.03).matched) == 1
    with pytest.raises(ValueError):
        synchronize_events(ev, td.frames, tolerance_s=-1)


def test_sync_flags_frame_mismatch_and_duplicates(td, metrica_files):
    ev = read_events(metrica_files[2]).iloc[[0]].copy()
    ev["start_frame"] = 99
    assert synchronize_events(ev, td.frames).matched["frame_mismatch"].iloc[0]
    dup = td.frames.copy()
    dup.loc[2, "time_s"] = dup.loc[1, "time_s"]
    res = synchronize_events(read_events(metrica_files[2]).iloc[[0]], dup)
    assert "ambiguous" in res.unmatched["reason"].iloc[0]


def test_validation_clean_data_flags_missing_only(td):
    rep = validate_tracking(td)
    assert rep.ok
    assert rep.get("player_ids").status == OK
    assert rep.get("missing_values").status == WARN and "ball missing in 1/5" in rep.get("missing_values").detail
    assert rep.get("duplicate_frames").status == OK
    assert rep.get("coordinates").status == OK
    assert rep.get("timestamps").status == OK and "25.00 Hz" in rep.get("timestamps").detail
    assert "Result: PASS" in rep.to_text()


def test_validation_detects_problems(td):
    f = td.frames
    f.loc[1, td.col("Home", "Player2", "x")] = 200.0       # far outside pitch + speed jump
    f.loc[4, "time_s"] = f.loc[3, "time_s"] - 0.04          # non-increasing time within period
    f.loc[1, "ball_x"] = np.inf
    f.loc[2, "frame"] = 10                                  # gap in frames
    rep = validate_tracking(td)
    assert not rep.ok
    assert rep.get("coordinates").status == FAIL and "1 infinite" in rep.get("coordinates").detail
    assert rep.get("timestamps").status == FAIL
    assert rep.get("continuity").status == WARN and "1 frame gaps" in rep.get("continuity").detail
    assert "players 1" in rep.get("continuity").detail  # jump out counted; return step spans the frame gap so is excluded


def test_validation_duplicates_and_missing_ids(td):
    td.frames.loc[1, "frame"] = td.frames.loc[0, "frame"]
    td.players.loc[0, "id_missing"] = True
    rep = validate_tracking(td)
    assert rep.get("duplicate_frames").status == FAIL
    assert rep.get("player_ids").status == FAIL


def test_missing_header_id_is_flagged(tmp_path, metrica_files):
    txt = metrica_files[0].read_text().replace("Player3", "", 1)
    h = tmp_path / "h.csv"
    h.write_text(txt)
    td = read_tracking(h, metrica_files[1])
    assert td.players["id_missing"].sum() == 1
    assert validate_tracking(td).get("player_ids").status == FAIL


def test_plot_smoke(td, metrica_files, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src.visualization.pitch import plot_event, plot_frame
    ax = plot_frame(td, 0, attacking_team="Away")
    ax.figure.savefig(tmp_path / "a.png")
    ev = read_events(metrica_files[2]).iloc[0].to_dict()
    plot_event(td, ev, 1, "Away").figure.savefig(tmp_path / "b.png")
    plot_frame(td, 2).figure.savefig(tmp_path / "c.png")  # ball missing
    assert (tmp_path / "c.png").stat().st_size > 0
