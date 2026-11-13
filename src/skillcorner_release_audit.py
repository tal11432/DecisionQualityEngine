"""SkillCorner release-frame feasibility audit on ONE match slice (no model, no features, no pipeline change).

    python -m src.skillcorner_release_audit --events <dynamic_events.csv> --tracking <tracking_prefix.jsonl> --out reports/skillcorner_release_results.txt

Release is detected from TRACKING ONLY (ball-passer distance). The event frames are used only to locate a +-2 s search window around the
pass; they never select the detected frame (passes with several candidate departures are ambiguous, not resolved with event frames).
All criteria below were fixed before the first measurement.

Departure rule (10 Hz): inside(t) = |ball - passer| <= 1.5 m. A departure at k needs inside(k-1) and NOT inside(k), inside(k+1), inside(k+2)
(look-ahead 2 frames = 0.2 s <= the 0.24 s Scenario-B window). Release frame r = k-1 (last frame in contact). It counts only if passer and ball
are DETECTED (not extrapolated) in frames k-1..k+2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HZ = 10.0
IN_M = 1.5
WINDOW = 20                # frames either side of the event span used to LOCATE the pass (not to choose the answer)
LOOKAHEAD = 2
CRITERIA = {
    "C1 identifiable release share >= 80%": 0.80,
    "C2 share with |release - event frame| <= 1 frame >= 80% (for frame_end or frame_start)": 0.80,
    "C3 |identifiable share completed - unsuccessful| <= 10 pp": 0.10,
    "C4 zero-duration events: identifiable share within 10 pp of the others and median |error| <= 1 frame": 0.10,
    "C5 same release frame with look-ahead 1 vs 2 frames in >= 90% of identified passes": 0.90,
    "C6 >= 80% of identified departures pass-like (>= 5 m/s over 0.3 s) with no sharp deflection": 0.80,
}
PASS_LIKE_MS, DEFLECT_DEG = 5.0, 60.0


def load_tracking(path: Path) -> dict:
    lines = Path(path).read_bytes().split(b"\n")[:-1]                  # last line of a byte-range prefix is truncated
    fr = [json.loads(l) for l in lines]
    n = len(fr)
    ball = np.full((n, 2), np.nan)
    ball_det = np.zeros(n, bool)
    players: dict[int, list] = {}
    frame_ids = np.array([f["frame"] for f in fr])
    for i, f in enumerate(fr):
        b = f["ball_data"]
        if b and b.get("x") is not None:
            ball[i] = (b["x"], b["y"]); ball_det[i] = bool(b.get("is_detected"))
        for p in f["player_data"]:
            a = players.setdefault(p["player_id"], [np.full((n, 2), np.nan), np.zeros(n, bool)])
            a[0][i] = (p["x"], p["y"]); a[1][i] = bool(p["is_detected"])
    if not (np.diff(frame_ids) == 1).all():
        raise ValueError("tracking frames are not consecutive")
    return {"frame0": int(frame_ids[0]), "n": n, "ball": ball, "ball_det": ball_det, "players": players,
            "first_play": int(next(f["frame"] for f in fr if f["player_data"]))}


def departures(d: np.ndarray, lookahead: int = LOOKAHEAD) -> list[int]:
    """Indices k (into d) where the ball leaves the passer: inside at k-1, outside at k..k+lookahead."""
    ins = d <= IN_M
    out = []
    for k in range(1, len(d) - lookahead):
        if np.isnan(d[k - 1:k + lookahead + 1]).any():
            continue
        if ins[k - 1] and not ins[k:k + lookahead + 1].any():
            out.append(k)
    return out


def analyse_pass(trk: dict, e: pd.Series, lookahead: int = LOOKAHEAD) -> dict:
    off = trk["frame0"]
    a, b = int(e.frame_start) - WINDOW - off, int(e.frame_end) + WINDOW - off
    res = {"event_idx": e.name, "reason": None}
    pl = trk["players"].get(int(e.player_id))
    if pl is None or a < 0 or b + 1 >= trk["n"]:
        res["reason"] = "passer not tracked or window outside the slice"; return res
    idx = np.arange(a, b + 1)
    d = np.hypot(*(trk["ball"][idx] - pl[0][idx]).T)
    res["det_share_window"] = float((pl[1][idx] & trk["ball_det"][idx]).mean())
    ks = departures(d, lookahead)
    res["n_departures"] = len(ks)
    if len(ks) == 0:
        res["reason"] = "no departure in window (ball never within 1.5 m of the passer, or never leaves)"; return res
    if len(ks) > 1:
        res["reason"] = "ambiguous: several departures in window"; return res
    k = ks[0]
    kk = idx[k]
    need = np.arange(kk - 1, kk + lookahead + 1)
    if not (pl[1][need].all() and trk["ball_det"][need].all()):
        res["reason"] = "passer or ball extrapolated (not detected) around the departure"; return res
    r = kk - 1
    res.update(identified=True, release_frame=int(r + off), err_end=int(r + off - e.frame_end), err_start=int(r + off - e.frame_start))
    ball = trk["ball"]
    res["speed_03s"] = float(np.hypot(*(ball[r + 3] - ball[r])) / 0.3) if r + 3 < trk["n"] else np.nan
    v1, v2 = ball[r + 1] - ball[r], (ball[r + 4] - ball[r + 1]) if r + 4 < trk["n"] else np.full(2, np.nan)
    cosang = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    res["deflect_deg"] = float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))) if np.isfinite(cosang) else np.nan
    # hold phase: frames in contact before the departure (arrival -> departure); 0-1 frames = release and reception not separable at 10 Hz
    ins = d <= IN_M
    j = k - 1
    while j - 1 >= 0 and ins[j - 1]:
        j -= 1
    res["contact_frames_before_release"] = int(k - 1 - j)
    return res


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    L = []
    say = L.append
    trk = load_tracking(Path(a.tracking))
    ev = pd.read_csv(a.events, low_memory=False)
    pp = ev[(ev.event_type == "player_possession") & (ev.end_type == "pass") & ev.pass_outcome.isin(["successful", "unsuccessful"])].copy()
    last = trk["frame0"] + trk["n"] - 1
    pp = pp[(pp.frame_start - WINDOW >= trk["first_play"]) & (pp.frame_end + WINDOW + 4 <= last)]
    pp["zero_duration"] = pp.frame_start == pp.frame_end
    say(f"slice: tracking frames {trk['frame0']}..{last} ({trk['n']} frames, 10 Hz; play from frame {trk['first_play']}); passes with a full window inside the slice: {len(pp)} "
        f"(successful {(pp.pass_outcome == 'successful').sum()}, unsuccessful {(pp.pass_outcome == 'unsuccessful').sum()}; zero-duration events {int(pp.zero_duration.sum())})")
    det = np.concatenate([v[1][trk['first_play']:] for v in trk["players"].values()])
    say(f"detection: ball detected in {trk['ball_det'][trk['first_play']:].mean():.1%} of frames; player positions detected in {det.mean():.1%} (the rest are extrapolated)")
    R = pd.DataFrame([analyse_pass(trk, e) for _, e in pp.iterrows()]).set_index("event_idx")
    R = R.join(pp[["pass_outcome", "zero_duration", "frame_start", "frame_end"]])
    R["identified"] = R["identified"].fillna(False).astype(bool)
    n = len(R); k = int(R.identified.sum())
    lo, hi = wilson(k, n)
    say(f"\n[Q1/Q2] identifiable release (unique departure, passer+ball detected): {k}/{n} = {k / n:.1%} (95% Wilson {lo:.1%}-{hi:.1%})")
    say("non-identified reasons: " + str(R[~R.identified].reason.value_counts().to_dict()))
    I = R[R.identified]
    say("\n[Q3] timing of the tracking-detected release relative to the event frames (frames; 1 frame = 0.1 s):")
    for fld in ("err_end", "err_start"):
        v = I[fld]
        say(f"  vs {fld[4:]}: median {v.median():+.0f}, IQR {v.quantile(.25):+.0f}..{v.quantile(.75):+.0f}, mean |e| {v.abs().mean():.2f}; share |e|<=1 frame {(v.abs() <= 1).mean():.1%}; share |e|=0 {(v == 0).mean():.1%}; "
            f"distribution {v.value_counts().sort_index().to_dict()}")
    say("\n[Q4] zero-duration events (frame_start == frame_end) vs others:")
    for zd, g in R.groupby("zero_duration"):
        gi = g[g.identified]
        say(f"  zero_duration={zd}: n={len(g)}, identifiable {len(gi)}/{len(g)} = {len(gi) / len(g):.1%}; median err_end {gi.err_end.median() if len(gi) else float('nan'):+.0f}, share |err_end|<=1 {(gi.err_end.abs() <= 1).mean() if len(gi) else float('nan'):.1%}")
    say("\n[outcome] completed vs unsuccessful:")
    for o, g in R.groupby("pass_outcome"):
        gi = g[g.identified]
        say(f"  {o}: n={len(g)}, identifiable {len(gi)}/{len(g)} = {len(gi) / len(g):.1%}; median err_end {gi.err_end.median():+.0f}; share |err_end|<=1 {(gi.err_end.abs() <= 1).mean():.1%}; "
            f"ambiguous (several departures) {(g.reason.fillna('').str.startswith('ambiguous')).mean():.1%}; extrapolated {(g.reason.fillna('').str.contains('extrapolated')).mean():.1%}")
    say("\n[why not identifiable] share of window frames with passer AND ball detected (not extrapolated), median [IQR], by status:")
    R["status"] = np.where(R.identified, "identified", R.reason.fillna("").str.split(":").str[0].str.split(" \\(").str[0])
    for st, g in R.groupby("status"):
        say(f"  {st}: n={len(g)}, detected share {g.det_share_window.median():.2f} [{g.det_share_window.quantile(.25):.2f}..{g.det_share_window.quantile(.75):.2f}]")
    for o, g in R.groupby("pass_outcome"):
        say(f"  detected share by outcome {o}: median {g.det_share_window.median():.2f}")
    say("[ambiguous examples] (several departures in the window)")
    for idx, r in R[R.reason.fillna("").str.startswith("ambiguous")].head(5).iterrows():
        say(f"  event {idx}: frames {int(r.frame_start)}-{int(r.frame_end)}, {r.pass_outcome}, departures {int(r.n_departures)}, detected share {r.det_share_window:.2f}")
    # the Metrica headline measure (ball-passer distance at the event frame), at frame_end, on frames where passer and ball are detected
    off = trk["frame0"]
    d_end, d_end_det = [], []
    for _, e in pp.iterrows():
        pl_ = trk["players"].get(int(e.player_id))
        i = int(e.frame_end) - off
        if pl_ is None or not (0 <= i < trk["n"]):
            continue
        dd = float(np.hypot(*(trk["ball"][i] - pl_[0][i])))
        d_end.append(dd)
        if pl_[1][i] and trk["ball_det"][i]:
            d_end_det.append(dd)
    d_end, d_end_det = np.array(d_end), np.array(d_end_det)
    say(f"\n[Metrica-comparable] ball-passer distance at frame_end: all passes n={len(d_end)}: median {np.median(d_end):.2f} m, share > 3 m {(d_end > 3).mean():.1%}; "
        f"detected passer+ball only n={len(d_end_det)}: median {np.median(d_end_det):.2f} m, share > 3 m {(d_end_det > 3).mean():.1%} "
        f"(Metrica G1 at its event frame: median 0.09 m, share > 3 m 1.0%; IDSSE: 2.4 m, 47.8%)")
    say("\n[Q5] release vs other ball interactions (identified departures):")
    pl = (I.speed_03s >= PASS_LIKE_MS) & (I.deflect_deg < DEFLECT_DEG)
    say(f"  pass-like speed (>= {PASS_LIKE_MS} m/s over 0.3 s): {(I.speed_03s >= PASS_LIKE_MS).mean():.1%}; sharp deflection (> {DEFLECT_DEG:.0f} deg within 0.5 s): {(I.deflect_deg >= DEFLECT_DEG).mean():.1%}; "
        f"pass-like and no deflection: {pl.mean():.1%}; median speed {I.speed_03s.median():.1f} m/s")
    cf = I.contact_frames_before_release
    say(f"  frames in contact before release: median {cf.median():.0f}; 0-1 frames (reception and release not separable at 10 Hz): {(cf <= 1).mean():.1%}")
    say("\n[Q6] look-ahead needed: re-running the detection with look-ahead 1 frame (0.1 s):")
    same = 0; changed = 0
    for idx, row in I.iterrows():
        r1 = analyse_pass(trk, pp.loc[idx], lookahead=1)
        if r1.get("identified") and r1["release_frame"] == row.release_frame:
            same += 1
        else:
            changed += 1
    say(f"  identical release frame in {same}/{len(I)} = {same / len(I):.1%}; changed/lost {changed}. Look-ahead used by the main rule: {LOOKAHEAD} frames = 0.2 s (<= the 0.24 s Scenario-B window).")
    # criteria
    share = lambda g: len(g[g.identified]) / len(g)
    succ, uns = R[R.pass_outcome == "successful"], R[R.pass_outcome == "unsuccessful"]
    zd, nz = R[R.zero_duration], R[~R.zero_duration]
    zi = zd[zd.identified]
    res = {
        "C1": k / n >= .80,
        "C2": max((I.err_end.abs() <= 1).mean(), (I.err_start.abs() <= 1).mean()) >= .80 if len(I) else False,
        "C3": abs(share(succ) - share(uns)) <= .10,
        "C4": abs(share(zd) - share(nz)) <= .10 and (zi.err_end.abs().median() <= 1 if len(zi) else False),
        "C5": same / max(len(I), 1) >= .90,
        "C6": pl.mean() >= .80,
    }
    say("\n== PRE-REGISTERED CRITERIA ==")
    for (name, thr), (key, ok) in zip(CRITERIA.items(), res.items()):
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    go = all(res.values())
    say(f"\nVERDICT: {'ALL CRITERIA MET' if go else 'NOT ALL CRITERIA MET'}  (n = {n} passes from one 12 MB slice; no model, no features)")
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
