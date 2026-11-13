"""DFL/IDSSE adapter, frozen eligibility, labels and the release-frame gate logic on a tiny synthetic match (no real data)."""
import numpy as np
import pandas as pd
import pytest

from src.idsse_labels import compare, frozen_eligibility, label_a, label_b
from src.idsse_pilot import GATE, gate_checks, release_measure, summarize_release
from src.ingestion import dfl

HOME = [f"H{i}" for i in range(1, 6)] + ["H6"]       # H6 only plays the second half (substitute)
AWAY = [f"A{i}" for i in range(1, 6)]


def meta_xml(pitch=(105, 68)):
    pl = "".join(f'<Player PersonId="{p}" ShirtNumber="{i + 1}" Starting="true"/>' for i, p in enumerate(HOME[:5]))
    pl2 = "".join(f'<Player PersonId="{p}" ShirtNumber="{i + 20}" Starting="true"/>' for i, p in enumerate(AWAY))
    return f"""<?xml version="1.0"?><PutDataRequest><MatchInformation>
    <General MatchId="M1" HomeTeamId="CH" GuestTeamId="CA" KickoffTime="2022-10-15T11:00:00.000+00:00"/>
    <Environment PitchX="{pitch[0]}" PitchY="{pitch[1]}"/>
    <Teams><Team TeamId="CH" Role="home"><Players>{pl}<Player PersonId="H6" ShirtNumber="9" Starting="false"/></Players></Team>
    <Team TeamId="CA" Role="guest"><Players>{pl2}</Players></Team></Teams></MatchInformation></PutDataRequest>"""


def frame(n, t, x, y, extra=""):
    xs = f' X="{x}"' if x is not None else ""
    return f'<Frame N="{n}" T="{t}"{xs} Y="{y}" S="1.0" M="1"{extra}/>'


def positions_xml(p2_offset_s=1000.0):
    """P1: frames 10..13 at 40 ms; P2: frames 100..102 starting p2_offset_s later. H1 has a missing X in frame 11; the ball has no frame 12;
    H6 exists only in period 2."""
    def T(sec):                                    # UTC clock
        from datetime import datetime, timedelta, timezone
        return (datetime(2022, 10, 15, 11, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)).isoformat(timespec="milliseconds")
    out = ['<?xml version="1.0"?><PutDataRequest><Positions><MetaData Type="pitch-size"><PitchSize X="105.00" Y="68.00"/></MetaData>']
    for sec_name, frames, t0 in (("firstHalf", range(10, 14), 1.0), ("secondHalf", range(100, 103), 1.0 + p2_offset_s)):
        people = (HOME[:5] + (["H6"] if sec_name == "secondHalf" else []), AWAY)
        for team_id, group in (("CH", people[0]), ("CA", people[1])):
            for k, pid in enumerate(group):
                side = -1 if team_id == "CH" else 1
                if sec_name == "secondHalf":
                    side = -side
                out.append(f'<FrameSet GameSection="{sec_name}" MatchId="M1" TeamId="{team_id}" PersonId="{pid}">')
                for i, n in enumerate(frames):
                    x = side * (10 + 5 * k)
                    out.append(frame(n, T(t0 + 0.04 * i), None if (pid == "H1" and n == 11) else x, 3 * k))
                out.append("</FrameSet>")
        out.append(f'<FrameSet GameSection="{sec_name}" MatchId="M1" TeamId="BALL" PersonId="B">')
        for i, n in enumerate(frames):
            if sec_name == "firstHalf" and n == 12:
                continue                                           # ball record absent for this frame
            out.append(frame(n, T(t0 + 0.04 * i), 0.5, 0.5, ' Z="0.1" BallPossession="1" BallStatus="1"'))
        out.append("</FrameSet>")
    out.append("</Positions></PutDataRequest>")
    return "\n".join(out)


def events_xml():
    def ev(i, time, body, extra=""):
        return f'<Event EventId="E{i}" EventTime="{time}" X-Position="52.50" Y-Position="34.00" MatchId="M1"{extra}>{body}</Event>'
    t = lambda s: f"2022-10-15T11:00:{s}+00:00"
    return "<PutDataRequest>" + "".join([
        ev(1, t("01.000"), '<KickOff TeamLeft="CH" TeamRight="CA" GameSection="firstHalf"><Play Recipient="H2" Evaluation="successfullyCompleted" Team="CH" Player="H1"><Pass/></Play></KickOff>'),
        ev(2, t("01.040"), '<Play Recipient="H3" Evaluation="successfullyCompleted" Team="CH" Player="H2"><Pass/></Play>'),
        ev(3, t("01.080"), '<Play Recipient="A1" Evaluation="unsuccessful" Team="CH" Player="H3"><Pass/></Play>'),
        ev(4, t("01.120"), '<BallClaiming Type="InterceptedBall" Team="CA" Player="A1"/>'),
        ev(5, t("01.040"), '<Play Recipient="H4" Evaluation="successfullyCompleted" Team="CH" Player="H5"><Cross/></Play>'),
        ev(6, t("01.040"), '<ThrowIn Side="left" Team="CH"><Play Recipient="H4" Evaluation="successfullyCompleted" Team="CH" Player="H5"><Pass/></Play></ThrowIn>'),
        ev(7, t("01.040"), '<Play Recipient="H4" Evaluation="successfullyCompleted" Team="CH" Player="H5"><Pass Direction="throughBall"/></Play>'),
        ev(8, t("01.040"), "<Delete />"),
        ev(9, t("01.000"), '<Play Evaluation="unsuccessful" Team="CA" Player="A2"><Pass/></Play>'),
    ]) + "</PutDataRequest>"


@pytest.fixture
def files(tmp_path):
    (tmp_path / "m.xml").write_text(meta_xml()); (tmp_path / "p.xml").write_text(positions_xml()); (tmp_path / "e.xml").write_text(events_xml())
    return tmp_path / "p.xml", tmp_path / "m.xml", tmp_path / "e.xml"


def load(files):
    td, qa, t0 = dfl.read_tracking(files[0], files[1])
    meta = dfl.read_match_info(files[1])
    f = td.frames
    bounds = {int(p): (float(g.time_s.min()), float(g.time_s.max())) for p, g in f.groupby("period")}
    return td, qa, t0, meta, dfl.read_events(files[2], meta, t0, bounds)


def test_missing_values_stay_nan_and_no_frame_or_player_is_dropped(files):
    td, qa, *_ = load(files)
    f = td.frames
    assert len(f) == 7 and f["frame"].tolist() == [10, 11, 12, 13, 100, 101, 102]          # every frame kept, incl. the one with no ball record
    assert np.isnan(f["Home__H1__x"].iloc[1]) and f["Home__H1__y"].iloc[1] == 0 and qa.missing_xy == 1   # missing X -> NaN, not 0
    assert np.isnan(f["ball_x"].iloc[2]) and qa.frames_without_ball == 1
    assert td.players.player_id.tolist().count("H6") == 1 and f["Home__H6__x"].iloc[:4].isna().all() and f["Home__H6__x"].iloc[4:].notna().all()   # substitute NaN while off
    assert qa.metadata_players_never_tracked == [] and qa.persons_not_in_metadata == []


def test_time_comes_from_timestamps_not_from_frame_offsets(files):
    td, qa, t0, *_ = load(files)
    f = td.frames
    assert f["time_s"].iloc[0] == 0.0 and f["time_s"].iloc[3] == pytest.approx(0.12)
    assert f["time_s"].iloc[4] == pytest.approx(1000.0)               # frame 100 starts 1000 s later by its T; (100-10)/25 s = 3.6 s would be the frame-offset answer
    assert qa.t_mismatch_across_objects == 0


def test_orientation_from_kickoff_and_pitch_from_metadata(files):
    td, *_ = load(files)
    assert td.attack_dir == {("Home", 1): 1, ("Away", 1): -1, ("Home", 2): -1, ("Away", 2): 1}      # home on the left in half 1 -> attacks +x
    assert (td.pitch.length, td.pitch.width) == (105.0, 68.0)


def test_pitch_mismatch_raises(tmp_path, files):
    (tmp_path / "m2.xml").write_text(meta_xml(pitch=(100, 68)))
    with pytest.raises(ValueError, match="pitch"):
        dfl.read_tracking(files[0], tmp_path / "m2.xml", dfl.Pitch(105, 68))


def test_events_keep_raw_attributes_on_the_tracking_clock(files):
    *_, ev = load(files)
    assert len(ev) == 9 and ev.set_index("event_id").loc["E2", "time_s"] == pytest.approx(1.04 - 1.0)   # EventTime on the same clock as frame T
    r = ev.set_index("event_id").loc["E3"]
    assert r.team == "Home" and r.recipient == "A1" and '"Play.Evaluation": "unsuccessful"' in r.raw
    assert ev.set_index("event_id").loc["E6", "set_piece"] == "ThrowIn" and bool(ev.set_index("event_id").loc["E8", "is_delete"])
    assert ev.period.tolist()[:3] == [1, 1, 1]


def test_frozen_eligibility_rules(files):
    *_, ev = load(files)
    el = frozen_eligibility(ev).set_index("event_id")
    assert el.loc[["E1", "E6"], "excluded_reason"].str.startswith("set piece").all()           # KickOff and ThrowIn wrappers
    assert el.loc["E5", "excluded_reason"] == "cross" and el.loc["E7", "excluded_reason"] == "through ball"
    assert el.loc["E8", "excluded_reason"] == "deleted event"
    assert bool(el.loc["E2", "eligible"]) and bool(el.loc["E3", "eligible"]) and not bool(el.loc["E4", "eligible"])   # BallClaiming is not a pass


def test_label_definitions_are_independent_and_never_forced(files):
    td, qa, t0, meta, ev = load(files)
    el = frozen_eligibility(ev)
    team = dict(zip(td.players.player_id, td.players.team))
    la, lb = label_a(el, team).set_index("event_idx"), label_b(el, team).set_index("event_idx")
    idx = el.set_index("event_id").event_idx
    assert la.loc[idx["E2"], "outcome"] == "completed" and la.loc[idx["E3"], "outcome"] == "intercepted"
    assert lb.loc[idx["E3"], "outcome"] == "intercepted"                                        # opposing InterceptedBall follows
    assert pd.isna(la.loc[idx["E9"], "outcome"]) and "recipient" in la.loc[idx["E9"], "reason"]   # no recipient -> unknown, not a class
    assert pd.isna(lb.loc[idx["E9"], "outcome"])
    c = compare(la.reset_index(), lb.reset_index())
    assert c["both"] == 1 and c["only_A"] == 0 and c["only_B"] == 0 and c["jaccard"] == 1.0


def test_definition_b_requires_the_claim_before_another_play_event(files):
    td, qa, t0, meta, ev = load(files)
    ev = ev.copy()
    ev.loc[ev.event_id == "E4", "time_s"] += 20.0                                                 # claim far outside the association window
    el = frozen_eligibility(ev)
    team = dict(zip(td.players.player_id, td.players.team))
    la, lb = label_a(el, team).set_index("event_idx"), label_b(el, team).set_index("event_idx")
    i3 = el.set_index("event_id").loc["E3", "event_idx"]
    assert la.loc[i3, "outcome"] == "intercepted" and pd.isna(lb.loc[i3, "outcome"])


# ---- release-frame gate logic ----
def test_release_measure_distinguishes_aligned_from_misaligned_event_times():
    from tests.test_leakage import REL, make_td
    td = make_td()                                                     # passer H1 holds the ball until row REL, then the ball leaves at 12 m/s
    aligned = pd.DataFrame({"time_s": [REL * 0.04], "team": ["Home"], "from_player": ["H1"]})
    late = pd.DataFrame({"time_s": [REL * 0.04 + 1.5], "team": ["Home"], "from_player": ["H1"]})      # event 1.5 s after the release
    a, b = release_measure(td, aligned), release_measure(td, late)
    assert a.d0.iloc[0] < 0.5 and a.speed_after.iloc[0] > 8 and not bool(a.censored.iloc[0])
    assert b.d0.iloc[0] > 10 and bool(b.censored.iloc[0])


def test_gate_checks_pass_for_reference_like_stats_and_fail_for_the_idsse_pattern():
    ref = {"median_d0": 0.088, "share_d0_over_3m": 0.010, "tau_median": 0.12, "tau_q25": 0.12, "tau_q75": 0.16, "speed_after_median": 11.46}
    assert all(gate_checks(ref, ref).values())
    bad = {"median_d0": 2.4, "share_d0_over_3m": 0.478, "tau_median": -0.24, "tau_q25": -0.35, "tau_q75": -0.04, "speed_after_median": 5.47}
    res = gate_checks(bad, ref)
    assert not any(res.values()) and GATE["median_d0_max_m"] == 0.5          # criteria were fixed before the measurement
    one_bad = dict(ref, speed_after_median=5.0)
    assert sum(gate_checks(one_bad, ref).values()) == 4                       # a single failed criterion is visible
    assert summarize_release(pd.DataFrame({"d0": [0.1, 0.2], "tau_s": [0.12, 0.16], "censored": [False, False], "speed_after": [10.0, 12.0]}))["median_d0"] == pytest.approx(0.15)
