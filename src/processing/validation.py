"""Tracking-data validation and concise report."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.schema import TrackingData

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def get(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)

    def to_text(self) -> str:
        lines = ["Tracking validation report", "=" * 26]
        lines += [f"[{c.status:<4}] {c.name}: {c.detail}" for c in self.checks]
        lines.append(f"Result: {'PASS' if self.ok else 'FAIL'} "
                     f"({sum(c.status == WARN for c in self.checks)} warnings)")
        return "\n".join(lines)


def validate_tracking(td: TrackingData, cfg: dict | None = None) -> ValidationReport:
    v = {"coord_margin_m": 3.0, "max_player_speed_ms": 12.0, "max_ball_speed_ms": 45.0,
         "dt_tolerance_s": 0.001, **((cfg or {}).get("validation", {}))}
    f, rep = td.frames, ValidationReport()
    n = len(f)
    if n == 0:
        rep.add("non_empty", FAIL, "no frames")
        return rep

    # 1. player identifiers
    miss = td.players[td.players["id_missing"]]
    dup_ids = td.players.duplicated("player_id").sum()
    if len(miss) or dup_ids:
        rep.add("player_ids", FAIL, f"{len(miss)} columns without identifier, {dup_ids} duplicated ids")
    else:
        rep.add("player_ids", OK, f"{len(td.players)} players, all identified")

    # 2. missing values
    pcols = td.player_columns()
    xcols = [c for c in pcols if c.endswith("__x")]
    ball_miss = f["ball_x"].isna() | f["ball_y"].isna()
    p_miss = f[xcols].isna().mean()
    never = [c.rsplit("__", 1)[0] for c in p_miss.index[p_miss == 1.0]]
    detail = (f"ball missing in {ball_miss.sum()}/{n} frames ({ball_miss.mean():.1%}); "
              f"player-position missing {f[xcols].isna().to_numpy().mean():.1%} overall")
    if never:
        detail += f"; never tracked: {', '.join(never)}"
    rep.add("missing_values", WARN if (ball_miss.any() or p_miss.max() > 0) else OK, detail)

    # 3. duplicate frames
    dup_f = f.duplicated(["period", "frame"]).sum()
    dup_t = f.duplicated(["period", "time_s"]).sum()
    rep.add("duplicate_frames", FAIL if dup_f or dup_t else OK,
            f"{dup_f} duplicate (period, frame), {dup_t} duplicate (period, time)")

    # 4. coordinates
    allc = f[pcols + ["ball_x", "ball_y"]].to_numpy(float)
    nonfinite = int(np.isinf(allc).sum())
    hx, hy = td.pitch.length / 2 + v["coord_margin_m"], td.pitch.width / 2 + v["coord_margin_m"]
    xs = f[[c for c in pcols + ["ball_x"] if c.endswith("x")]].to_numpy(float)
    ys = f[[c for c in pcols + ["ball_y"] if c.endswith("y")]].to_numpy(float)
    with np.errstate(invalid="ignore"):
        oob = int((np.abs(xs) > hx).sum() + (np.abs(ys) > hy).sum())
    rep.add("coordinates", FAIL if nonfinite else (WARN if oob else OK),
            f"{nonfinite} infinite values; {oob} coordinates beyond pitch +{v['coord_margin_m']}m margin")

    # 5. timestamps
    t, fr, per = f["time_s"].to_numpy(float), f["frame"].to_numpy(float), f["period"].to_numpy(float)
    same = per[1:] == per[:-1]
    dt, dfr = np.diff(t)[same], np.diff(fr)[same]
    step = dt[dfr == 1]
    med = float(np.median(step)) if len(step) else float("nan")
    nonmono = int((dt <= 0).sum())
    irregular = int((np.abs(step - med) > v["dt_tolerance_s"]).sum()) if len(step) else 0
    inconsistent = int((np.abs(dt - dfr * med) > v["dt_tolerance_s"]).sum()) if len(step) else 0
    fps = f"{1 / med:.2f} Hz" if med == med and med > 0 else "n/a"
    rep.add("timestamps", FAIL if nonmono else (WARN if irregular or inconsistent else OK),
            f"rate {fps}; {nonmono} non-increasing times, {irregular} irregular steps, "
            f"{inconsistent} frame/time disagreements")

    # 6. continuity
    gaps = int((dfr > 1).sum())
    missing_frames = int((dfr[dfr > 1] - 1).sum())
    cons = (dfr == 1) & (dt > 0)
    jumps = {}
    if cons.any():
        for label, cols, lim in (("player", None, v["max_player_speed_ms"]), ("ball", ["ball"], v["max_ball_speed_ms"])):
            names = ["ball"] if cols else sorted({c.rsplit("__", 1)[0] for c in pcols})
            cnt = 0
            for nm in names:
                cx, cy = (f"{nm}_x", f"{nm}_y") if nm == "ball" else (f"{nm}__x", f"{nm}__y")
                dx, dy = np.diff(f[cx].to_numpy(float))[same], np.diff(f[cy].to_numpy(float))[same]
                with np.errstate(invalid="ignore", divide="ignore"):
                    sp = np.hypot(dx, dy)[cons] / dt[cons]
                cnt += int((sp > lim).sum())
            jumps[label] = cnt
    rep.add("continuity", WARN if gaps or any(jumps.values()) else OK,
            f"{gaps} frame gaps ({missing_frames} missing frames); implausible-speed jumps: "
            f"players {jumps.get('player', 0)}, ball {jumps.get('ball', 0)}")
    return rep
