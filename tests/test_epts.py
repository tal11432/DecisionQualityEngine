"""EPTS/FIFA reader (Sample Game 3 format) on a tiny synthetic match."""
import json

import numpy as np
import pandas as pd
import pytest

from src.ingestion.epts import apply_identity_swaps, check_events, detect_identity_swaps, read_events, read_metadata, read_tracking
from src.processing.sync import synchronize_events
from src.schema import Pitch

HOME = [f"P{i}" for i in range(1, 7)]       # P6 replaces P2 from frame 3
AWAY = [f"Q{i}" for i in range(1, 6)]
LEFT = [(0.05, 0.5), (0.2, 0.3), (0.2, 0.7), (0.4, 0.4), (0.4, 0.6)]


def meta_xml(home_dir="left_to_right", away_dir="right_to_left", size=(105, 68), second=(5, 6)):
    pl = "".join(f'<Player id="{p}" teamId="TA"><Name>x</Name><ShirtNumber>{i + 1}</ShirtNumber></Player>' for i, p in enumerate(HOME))
    pl += "".join(f'<Player id="{p}" teamId="TB"><Name>x</Name><ShirtNumber>{i + 20}</ShirtNumber></Player>' for i, p in enumerate(AWAY))
    ch = "".join(f'<PlayerChannel channelId="{a}" id="{p}_{a}" playerId="{p}"/>' for p in HOME + AWAY for a in "xy")

    def spec(s, e, home):
        ids = home + AWAY
        regs = "".join(f'<SplitRegister separator=","><PlayerChannelRef playerChannelId="{p}_x"/><PlayerChannelRef playerChannelId="{p}_y"/></SplitRegister>' for p in ids)
        return f'<DataFormatSpecification endFrame="{e}" separator=":" startFrame="{s}"><StringRegister name="frameCount"/><SplitRegister separator=";">{regs}</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>'
    specs = spec(1, 2, HOME[:5]) + spec(3, 6, ["P1", "P6", "P3", "P4", "P5"])
    prm = lambda n, v: f"<ProviderParameter><Name>{n}</Name><Value>{v}</Value></ProviderParameter>"
    return f"""<?xml version="1.0"?><main><Metadata><GlobalConfig><FrameRate>25</FrameRate><ProviderGlobalParameters>
    {prm('first_half_start', 1)}{prm('first_half_end', 4)}{prm('second_half_start', second[0])}{prm('second_half_end', second[1])}
    {prm('first_extra_half_start', '')}{prm('first_extra_half_end', '')}</ProviderGlobalParameters></GlobalConfig>
    <Sessions><Session id="s"><MatchParameters><Score idLocalTeam="TA" idVisitingTeam="TB"/></MatchParameters>
    <FieldSize><Width>{size[0]}</Width><Height>{size[1]}</Height></FieldSize></Session></Sessions>
    <Teams><Team id="TA"><ProviderTeamsParameters>{prm('attack_direction_first_half', home_dir)}</ProviderTeamsParameters></Team>
    <Team id="TB"><ProviderTeamsParameters>{prm('attack_direction_first_half', away_dir)}</ProviderTeamsParameters></Team></Teams>
    <Players>{pl}</Players><PlayerChannels>{ch}</PlayerChannels></Metadata>
    <DataFormatSpecifications>{specs}</DataFormatSpecifications></main>"""


def tracking_txt(break_line=False):
    lines = []
    for f in range(1, 7):
        h, a = (LEFT, [(1 - x, y) for x, y in LEFT]) if f <= 4 else ([(1 - x, y) for x, y in LEFT], LEFT)   # sides swap in half 2
        pos = [f"{x},{y}" for x, y in h] + [f"{x},{y}" for x, y in a]
        if f == 2:
            pos[0] = "NaN,NaN"                                        # untracked player must stay NaN
        ball = "NaN,NaN" if f == 3 else "0.5,0.5"
        lines.append(f"{f}:" + ";".join(pos) + ":" + ball)
    if break_line:
        lines[3] = lines[3].replace(";0.4,0.4", "")                    # ragged line
    return "\n".join(lines) + "\n"


def events_json():
    ev = lambda team, typ, sub, f, frm, to, sx, sy, ex, ey, per=1: {
        "team": {"id": team}, "type": {"name": typ}, "subtypes": sub, "period": per,
        "start": {"frame": f, "time": f / 25, "x": sx, "y": sy}, "end": {"frame": f + 1, "time": (f + 1) / 25, "x": ex, "y": ey},
        "from": None if frm is None else {"id": frm}, "to": None if to is None else {"id": to}}
    return json.dumps({"data": [
        ev("TA", "PASS", None, 2, "P1", "P3", 0.2, 0.3, 0.4, 0.4),
        ev("TB", "BALL LOST", [{"name": "HEAD"}, {"name": "INTERCEPTION"}], 4, "Q1", None, 0.9, 0.5, None, None),
        ev("TA", "PASS", {"name": "CROSS"}, 5, "P6", "P3", 0.5, 0.5, 0.6, 0.6, 2),
        ev("TA", "PASS", None, 3, "P99", "P3", 0.5, 0.5, 0.6, 0.6),      # unknown player id
    ]})


@pytest.fixture
def files(tmp_path):
    (tmp_path / "t.txt").write_text(tracking_txt())
    (tmp_path / "m.xml").write_text(meta_xml())
    (tmp_path / "e.json").write_text(events_json())
    return tmp_path / "t.txt", tmp_path / "m.xml", tmp_path / "e.json"


def test_metadata(files):
    m = read_metadata(files[1])
    assert m.fps == 25 and m.pitch_size == (105, 68) and m.periods == {1: (1, 4), 2: (5, 6)}
    assert m.team_side == {"TA": "Home", "TB": "Away"} and m.shirt["P6"] == "6"
    assert [(s, e) for s, e, _ in m.segments] == [(1, 2), (3, 6)] and m.segments[1][2][1] == "P6"


def test_tracking_parsing_slots_missing_values_and_clock(files):
    td = read_tracking(files[0], files[1], Pitch())
    f = td.frames
    assert len(f) == 6 and f["period"].tolist() == [1, 1, 1, 1, 2, 2] and f["frame"].tolist() == list(range(1, 7))
    assert f["time_s"].tolist() == pytest.approx([i / 25 for i in range(1, 7)])
    # substitution: P2 tracked only in frames 1-2, P6 only from frame 3 (slot 2 reused); never invented
    assert f["Home__P2__x"].notna().tolist() == [True, True] + [False] * 4
    assert f["Home__P6__x"].notna().tolist() == [False, False] + [True] * 4
    assert np.isnan(f["Home__P1__x"].iloc[1]) and np.isnan(f["ball_x"].iloc[2]) and f["ball_x"].iloc[0] == 0.0   # NaN preserved
    assert set(td.players.team) == {"Home", "Away"} and td.players.id_missing.sum() == 0 and td.players.jersey.tolist()[0] == "1"


def test_coordinate_conversion_matches_csv_convention(files):
    td = read_tracking(files[0], files[1], Pitch())
    assert td.frames["Home__P1__x"].iloc[0] == pytest.approx((0.05 - 0.5) * 105)     # same maths as metrica_to_metric
    assert td.frames["Home__P3__y"].iloc[0] == pytest.approx((0.5 - 0.7) * 68)       # y flipped to point up


def test_attack_direction_inferred_and_crosschecked(files):
    td = read_tracking(files[0], files[1], Pitch())
    assert td.attack_dir == {("Home", 1): 1, ("Away", 1): -1, ("Home", 2): -1, ("Away", 2): 1}
    files[1].write_text(meta_xml(home_dir="right_to_left", away_dir="left_to_right"))
    with pytest.raises(ValueError, match="contradicts metadata"):
        read_tracking(files[0], files[1], Pitch())


def test_explicit_errors_not_silent_repairs(files, tmp_path):
    files[1].write_text(meta_xml(size=(100, 60)))
    with pytest.raises(ValueError, match="pitch size"):
        read_tracking(files[0], files[1], Pitch())
    files[1].write_text(meta_xml(second=(5, 5)))                      # frame 6 belongs to no period
    with pytest.raises(ValueError, match="outside every period"):
        read_tracking(files[0], files[1], Pitch())
    files[1].write_text(meta_xml())
    (tmp_path / "ragged.txt").write_text(tracking_txt(break_line=True))
    with pytest.raises(ValueError, match="malformed line"):
        read_tracking(tmp_path / "ragged.txt", files[1], Pitch())
    with pytest.raises(FileNotFoundError):
        read_tracking(tmp_path / "none.txt", files[1], Pitch())
    with pytest.raises(FileNotFoundError):
        read_events(tmp_path / "none.json", files[1])


def test_events_schema_vocabulary_and_coordinates(files):
    ev = read_events(files[2], files[1], Pitch())
    assert list(ev.columns)[:6] == ["team", "type", "subtype", "period", "start_frame", "start_time_s"]
    assert ev.team.tolist() == ["Home", "Away", "Home", "Home"]
    assert ev.subtype.tolist()[1:3] == ["HEAD-INTERCEPTION", "CROSS"] and pd.isna(ev.subtype.iloc[0])   # list -> '-' joined, dict -> name
    assert ev.start_x.iloc[0] == pytest.approx(-31.5) and ev.start_y.iloc[0] == pytest.approx(13.6)
    assert np.isnan(ev.end_x.iloc[1]) and pd.isna(ev.to_player.iloc[1])                               # missing stays missing


def test_event_tracking_synchronization_and_consistency_checks(files):
    td = read_tracking(files[0], files[1], Pitch())
    ev = read_events(files[2], files[1], Pitch())
    res = synchronize_events(ev, td.frames, 0.02)
    assert res.matched["frame"].tolist() == [2, 4, 5, 3] and res.matched["frame_mismatch"].sum() == 0
    assert synchronize_events(ev.assign(start_time_s=ev.start_time_s + 0.01), td.frames, 0.005).matched.empty   # tolerance enforced
    chk = check_events(ev, td)
    assert chk["from_player_unknown"] == 1 and chk["from_player_wrong_team"] == 0 and chk["start_clock_mismatch"] == 0


def test_same_downstream_labels_as_csv_games(files):
    """The label rules of the completion pipeline apply unchanged to EPTS events."""
    from src.pass_completion_features import label_events
    lab = label_events(read_events(files[2], files[1], Pitch()))
    assert lab.outcome.tolist()[0] == "completed" and lab.outcome.tolist()[3] == "completed" and lab.outcome.isna().tolist() == [False, True, True, False]
    assert "non-ground interception" in lab.reason.iloc[1] and "subtype" in lab.reason.iloc[2]


# ---------------------------------------------------------------- identity swap detection / opt-in repair
def _swap_case(swap=True):
    from src.schema import TrackingData
    n = 60
    pos = {"H1": (-20.0, 0.0), "H2": (20.0, 0.0), "H3": (0.0, 20.0), "A1": (0.0, -20.0)}
    f = {"period": pd.array([1] * n, dtype="Int64"), "frame": pd.array(range(1, n + 1), dtype="Int64"), "time_s": np.arange(1, n + 1) / 25}
    for (t, p), (x, y) in {("Home", k): v for k, v in pos.items() if k != "A1"}.items():
        f[f"{t}__{p}__x"], f[f"{t}__{p}__y"] = np.full(n, x), np.full(n, y)
    f["Away__A1__x"], f["Away__A1__y"] = np.full(n, 0.0), np.full(n, -20.0)
    ev_frames = list(range(2, 50, 2))                                 # 24 events, alternating holders
    bx, by = np.zeros(n), np.zeros(n)
    rows, who = [], []
    for k, fr in enumerate(ev_frames):
        holder = "H1" if k % 2 == 0 else "H2"
        bx[fr - 1], by[fr - 1] = pos[holder]
        named = ({"H1": "H2", "H2": "H1"}[holder]) if swap else holder       # tracking slots swapped vs events
        rows.append(("Home", "PASS", None, 1, fr, fr / 25, fr + 1, (fr + 1) / 25, named, "H3", 0, 0, 0, 0))
    f["ball_x"], f["ball_y"] = bx, by
    pl = pd.DataFrame([("Home", p, None, False) for p in ("H1", "H2", "H3")] + [("Away", "A1", None, False)],
                      columns=["team", "player_id", "jersey", "id_missing"])
    ev = pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start_frame", "start_time_s", "end_frame", "end_time_s",
                                     "from_player", "to_player", "start_x", "start_y", "end_x", "end_y"])
    return TrackingData(pd.DataFrame(f), pl, Pitch(), "x"), ev


def test_identity_swap_detected_only_when_systematic_and_mutual():
    td, ev = _swap_case(swap=True)
    s = detect_identity_swaps(td, ev)
    assert len(s) == 1 and (s[0]["a"], s[0]["b"]) == ("H1", "H2") and s[0]["rate_a_at_b"] == 1.0 and s[0]["n_a"] == 12
    td2, ev2 = _swap_case(swap=False)
    assert detect_identity_swaps(td2, ev2) == []
    assert detect_identity_swaps(td, ev.iloc[:8]) == []                  # too few events: never inferred from a handful


def test_apply_identity_swaps_is_opt_in_and_repairs_consistency():
    td, ev = _swap_case(swap=True)
    fixed = apply_identity_swaps(td, detect_identity_swaps(td, ev))
    assert detect_identity_swaps(fixed, ev) == [] and "identity_swaps_applied" in fixed.source
    assert td.frames["Home__H1__x"].iloc[0] == -20.0                      # original object untouched
    assert fixed.frames["Home__H1__x"].iloc[0] == 20.0
