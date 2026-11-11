"""Small synthetic Metrica-format fixtures (not real data)."""
import numpy as np
import pytest

NAN = "NaN"


def _team_csv(path, team, players, rows):
    """players: list of (jersey, id); rows: list of (period, frame, time, [(x,y)...], (bx,by))."""
    n = len(players)
    l1 = ["", "", ""] + sum([[team, ""] for _ in range(n)], []) + [team, ""]
    l2 = ["", "", ""] + sum([[str(j), ""] for j, _ in players], []) + ["", ""]
    l3 = ["Period", "Frame", "Time [s]"] + sum([[pid, ""] for _, pid in players], []) + ["Ball", ""]
    lines = [",".join(l1), ",".join(l2), ",".join(l3)]
    for per, fr, t, pos, ball in rows:
        cells = [str(per), str(fr), str(t)]
        for x, y in pos + [ball]:
            cells += [str(x), str(y)]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def metrica_files(tmp_path):
    # Home: 5 players spread on the left half; Away: 5 on the right half at kick-off (period 1).
    # Period 2 starts after a side swap (teams stand on mirrored halves).
    hp = [(str(i), f"Player{i}") for i in range(1, 6)]
    ap = [(str(i), f"Player{i}") for i in range(11, 16)]
    left = [(0.05, 0.5), (0.2, 0.3), (0.2, 0.7), (0.4, 0.4), (0.4, 0.6)]
    right = [(1 - x, y) for x, y in left]
    frames = [(1, 1, 0.04), (1, 2, 0.08), (1, 3, 0.12), (2, 4, 2700.00), (2, 5, 2700.04)]
    hrows, arows = [], []
    for k, (per, fr, t) in enumerate(frames):
        h, a = (left, right) if per == 1 else (right, left)
        h, a = [list(p) for p in h], [list(p) for p in a]
        if k == 1:
            h[0] = ["NaN", "NaN"]  # missing player
        ball = (0.5, 0.5) if k != 2 else ("NaN", "NaN")  # missing ball
        hrows.append((per, fr, t, [tuple(p) for p in h], ball))
        arows.append((per, fr, t, [tuple(p) for p in a], ball))
    hf, af, ef = tmp_path / "home.csv", tmp_path / "away.csv", tmp_path / "events.csv"
    _team_csv(hf, "Home", hp, hrows)
    _team_csv(af, "Away", ap, arows)
    ef.write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
        "Home,PASS,,1,2,0.08,3,0.12,Player2,Player3,0.2,0.3,0.4,0.4\n"
        "Home,PASS,,1,2,0.081,3,0.12,Player2,Player3,0.2,0.3,0.4,0.4\n"   # within tol
        "Away,SHOT,,1,9,5.0,9,5.1,Player11,,0.9,0.5,1.0,0.5\n"             # no frame in tol
        "Away,PASS,,3,9,5.0,9,5.1,Player11,,0.9,0.5,1.0,0.5\n"             # period w/o tracking
        "Away,BALL LOST,,1,,,,,Player11,,NaN,NaN,NaN,NaN\n"                # missing time
        "Home,PASS,,2,4,2700.0,5,2700.04,Player2,Player3,0.2,0.3,0.4,0.4\n"
    )
    return hf, af, ef
