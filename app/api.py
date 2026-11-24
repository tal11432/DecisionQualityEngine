"""Scoring API of the local Decision Score app: a hand-marked situation (StatsBomb units, 120 x 80, the team in possession attacks toward x = 120) -> the scored options.
Uses the same models and code as the research (research/dq/dscore): shot from x >= 60, ground passes of 3-40 m to the marked teammates (offside by the marked opponents), marked assumption A1 on."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "dq"))
from dscore import score as S, core
from dscore.core import Situation

def _pt(p, name):
    try: x, y = float(p[0]), float(p[1])
    except Exception: raise ValueError(f"bad point for {name}")
    if not (0 <= x <= 120 and 0 <= y <= 80): raise ValueError(f"{name} is outside the pitch")
    return [x, y]

def score_situation(payload):
    a = payload.get("carrier")
    if a is None: raise ValueError("place the ball carrier")
    a = _pt(a, "carrier"); mates = [_pt(m, "teammate") for m in payload.get("teammates", [])]; opps = [_pt(o, "opponent") for o in payload.get("opponents", [])]
    gk = _pt(payload["keeper"], "keeper") if payload.get("keeper") else None
    O = opps + ([gk] if gk else [])
    if not O: raise ValueError("place at least one opponent (or the opposing keeper)")
    thr = float(payload.get("threshold", 26)); top = max(1, min(int(payload.get("top", 10)), 10))
    s = Situation(a, np.array(O, float).reshape(-1, 2), [np.array(m, float) for m in mates], gk)
    opts = S.all_options(s); zone = core.zone(a[0]); warnings = []
    if zone != "attacking": warnings.append(f"the ball carrier is in the {zone} zone: in this zone the score is not validated (differences between options are thousandths of xG); read the order, not the numbers")
    if a[0] < core.SHOT_MIN_X: warnings.append("no shot option below x = 60")
    if gk is None: warnings.append("no opposing keeper marked: assumed at (118, 40) for the shot")
    if len(O) < 8: warnings.append(f"only {len(O)} opponents marked: with few opponents the offside line and the pass risk are less reliable")
    out, offside, turnover = [], 0, None
    for o in opts:
        if o["kind"] == "turnover": turnover = dict(score=o["score"], q=o["v"], counter_assumed=bool(o.get("counter_assumed"))); continue
        if o["status"] == "certain offside": offside += 1; continue
        row = dict(kind=o["kind"], score=int(o["score"]), q=float(o["v"]), status=o["status"], counter_assumed=bool(o.get("counter_assumed", False)))
        if o["kind"] == "shot":
            f = core.shot_features(s); row.update(target=[120.0, 40.0], xg=float(o["xg"]), cont=float(o["cont"]), dist_m=float(f["dist"] * 105 / 120), angle_deg=float(np.degrees(f["angle"])), n_tri=int(f["n_tri"]), gk_m=float(f["gk_d"] * 105 / 120))
        else: row.update(target=[float(o["target"][0]), float(o["target"][1])], xp=float(o["xp"]), chain=float(o["xa"]), dist_m=float(o["d"]), loss=float(o["loss"]))
        out.append(row)
    out.sort(key=lambda r: -r["q"]); kept = [r for r in out if r["score"] >= thr][:top]
    for i, r in enumerate(kept): r["rank"] = i + 1
    return dict(options=kept, n_options=len(out), n_offside=offside, turnover=turnover, zone=zone, warnings=warnings, threshold=thr)
