"""Is a SET PIECE followed by a PASS one action or two? Event-convention and tracking spot checks (Metrica Sample Games 1-3).

Usage: python research/set_piece_check/spot_checks.py        (needs data/metrica for games 1-3; prints, writes nothing)
Independent of the proposed filter: it defines its own mask, and restores the legacy label population if the proposed
``exclude_set_piece_deliveries`` argument exists. Diagnostic only; no model is fitted."""
import inspect, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import src.pass_completion_features as F
from src.analytics.feasibility import FeasibilityParams
from src.cli import _load
from src.config import load_config
from src.pass_dataset import DecisionContract
from src.processing.sync import synchronize_events

WINDOW_S = 1.0
cfg = load_config(); TOL = cfg["sync"]["tolerance_s"]
CONTRACT, PARAMS = DecisionContract.from_config(cfg), FeasibilityParams.from_config(cfg)
_orig = F.label_events
if "exclude_set_piece_deliveries" in inspect.signature(_orig).parameters:
    F.label_events = lambda e: _orig(e, exclude_set_piece_deliveries=False)      # legacy population, as in the committed reports


def delivery_mask(ev, window=WINDOW_S):
    prev = ev.shift(1)
    return (ev.type.isin(["PASS", "BALL LOST"]) & (prev.type == "SET PIECE") & (prev.team == ev.team) & (prev.from_player == ev.from_player)
            & (prev.period == ev.period) & ((ev.start_time_s - prev.end_time_s) <= window))


print("== 1. EVENT STRUCTURE (does the SET PIECE marker and the delivery coincide?) ==")
rows = []
for g in (1, 2, 3):
    td, ev = _load(cfg, g); nxt = ev.shift(-1); sp = ev.type == "SET PIECE"
    follow = sp & nxt.type.isin(["PASS", "BALL LOST", "SHOT"]) & (nxt.team == ev.team) & (nxt.from_player == ev.from_player) & ((nxt.start_time_s - ev.end_time_s) <= WINDOW_S)
    gap = (nxt.start_time_s - ev.end_time_s)[follow]
    print(f"G{g}: SET PIECE events {int(sp.sum())}; max |marker duration| {(ev.end_time_s - ev.start_time_s)[sp].abs().max():.2f}s; followed (same team, same player, <= {WINDOW_S}s) by PASS/BALL LOST/SHOT: {int(follow.sum())}; "
          f"gap median {gap.median():.2f}s, share exactly 0: {(gap == 0).mean():.0%}, max {gap.max():.2f}s; other followers: {nxt[sp & ~follow].type.value_counts().to_dict()}")

print("\n== 2. SCENARIO-A ELIGIBLE POPULATION (legacy labels) ==")
tabs = []
for g in (1, 2, 3):
    td, ev = _load(cfg, g)
    t = F.build_features(td, ev, "A", TOL, CONTRACT, PARAMS).table
    m = delivery_mask(ev); prev = ev.shift(1)
    t = t.assign(game=g, sp=m.loc[t.event_idx].to_numpy(), sub=prev.loc[t.event_idx, "subtype"].to_numpy())
    tabs.append(t)
d = pd.concat(tabs, ignore_index=True)
print(f"eligible n={len(d)}; deliveries {int(d.sp.sum())} ({d.sp.mean():.1%}): completed {int(d[d.sp].y_completed.sum())}, intercepted {int((1 - d[d.sp].y_completed).sum())}")
print("by restart subtype:", d[d.sp]["sub"].value_counts().to_dict())
print("interception rate: deliveries %.3f, open play %.3f" % (1 - d[d.sp].y_completed.mean(), 1 - d[~d.sp].y_completed.mean()))

print("\n== 3. TRACKING SPOT CHECKS at the event frame (all pass-related candidates with labels) ==")
rows = []
for g in (1, 2, 3):
    td, ev = _load(cfg, g); T = F._Track.of(td); L, W = td.pitch.length, td.pitch.width
    lab = _orig(ev) if "exclude_set_piece_deliveries" not in inspect.signature(_orig).parameters else _orig(ev, exclude_set_piece_deliveries=False)
    cand = lab[lab.outcome.notna()].event_idx; e = ev.loc[cand]; m = delivery_mask(ev).loc[cand]; prev = ev.shift(1).loc[cand]
    rel = synchronize_events(e[["team", "period", "start_time_s", "from_player"]], td.frames, TOL, "start_time_s", frame_col=None).matched.set_index("event_idx")["frame_row"].to_dict()
    for idx in cand:
        if idx not in rel or rel[idx] < 30 or rel[idx] + 30 >= T.n: continue
        r = rel[idx]; j = [k for k in range(len(T.pid)) if T.team[k] == e.loc[idx, "team"] and T.pid[k] == e.loc[idx, "from_player"]]
        if not j: continue
        j = j[0]
        rows.append(dict(game=g, sp=bool(m[idx]), sub=prev.loc[idx, "subtype"] if m[idx] else "open play", bx=T.bx[r], by=T.by[r],
                         dist=np.hypot(T.bx[r] - T.X[r, j], T.by[r] - T.Y[r, j]), nan_prev25=int((~np.isfinite(T.bx[r - 25:r])).sum()),
                         edge=min(L / 2 - abs(T.bx[r]), W / 2 - abs(T.by[r])),
                         v_after=np.hypot(T.bx[r + 10] - T.bx[r], T.by[r + 10] - T.by[r]) / 0.4))
t = pd.DataFrame(rows)
s = t[t.sp]
print("                                   deliveries   open play")
print(f"n                                  {len(s):10d} {len(t) - len(s):11d}")
print(f"ball untracked in all 25 frames before event  {(s.nan_prev25 >= 25).mean():6.1%} {(t[~t.sp].nan_prev25 >= 25).mean():10.1%}")
print(f"tracked ball coincides with passer (<1 cm)    {(s.dist < 0.01).mean():6.1%} {(t[~t.sp].dist < 0.01).mean():10.1%}")
print(f"median ball speed over next 0.4 s (m/s)       {s.v_after.median():6.1f} {t[~t.sp].v_after.median():10.1f}")
print("ball position by restart type (median x, y; share on the expected spot):")
ti = s[s["sub"] == "THROW IN"]; ck = s[s["sub"] == "CORNER KICK"]; ko = s[s["sub"].fillna("").str.startswith("KICK OFF")]
print(f"  THROW IN  n={len(ti)}: ball within 1 m of a touch/goal line {(ti.edge < 1.0).mean():.0%} (open play: {(t[~t.sp].edge < 1.0).mean():.1%})")
dc = np.hypot(52.5 - ck.bx.abs(), 34 - ck.by.abs())
print(f"  CORNER    n={len(ck)}: distance ball -> nearest corner flag: median {dc.median():.1f} m, max {dc.max():.1f} m (taker stands a few metres off the flag)")
print(f"  KICK OFF  n={len(ko)}: ball within 1.5 m of the centre spot {(np.hypot(ko.bx, ko.by) < 1.5).mean():.0%}")
