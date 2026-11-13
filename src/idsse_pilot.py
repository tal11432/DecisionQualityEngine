"""One-match IDSSE pilot (J03WPY): quality checks, release-frame gate, label comparison, compatibility run.

    python -m src.idsse_pilot --out reports/idsse_pilot_results.txt

Gates run in order and the run STOPS at the first critical failure (release-frame gate; label agreement).
No model is trained or evaluated. Existing Metrica code and results are only read (the Metrica Game 1 reference measurement).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion import dfl
from src.ingestion.epts import detect_identity_swaps
from src.processing.sync import synchronize_events
from src.processing.validation import validate_tracking
from src.idsse_labels import JACCARD_MIN, compare, frozen_eligibility, label_a, label_b

MID = "J03WPY"
DIR = Path("data/idsse") / MID
FILES = {"meta": f"DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-{MID}.xml", "events": f"DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-{MID}.xml",
         "positions": f"DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-{MID}.xml"}

# ---- release-frame gate: criteria fixed BEFORE the measurement was run ----
GATE = {"median_d0_max_m": 0.5, "share_d0_over_3m_max": 0.03, "median_tau_diff_max_s": 0.08, "iqr_ratio_max": 2.0, "median_speed_after_min_ms": 8.0}
RELEASE_DIST_M = 1.0       # the ball has "left the passer" when it is farther than this
PRE_WINDOW = 12            # frames before the event searched for the release (0.48 s)
POST_FRAMES = 6            # speed window after the event frame (0.24 s)


def release_measure(td, ev: pd.DataFrame) -> pd.DataFrame:
    """Per event: ball-passer distance at the nearest frame (d0), release lag tau (s, first time the ball is > 1 m from the passer,
    searched from -0.48 s; ``censored`` if already farther at the start of the window) and mean ball speed over the next 6 frames.
    ``ev`` needs columns time_s, team, from_player. Identical code is used for IDSSE and for the Metrica reference."""
    f = td.frames
    t = f["time_s"].to_numpy(float)
    per = f["period"].to_numpy(dtype=float, na_value=np.nan)
    fr = f["frame"].to_numpy(dtype=float, na_value=np.nan)
    bx, by = f["ball_x"].to_numpy(float), f["ball_y"].to_numpy(float)
    out = []
    for _, e in ev.iterrows():
        j = int(np.searchsorted(t, e["time_s"]))
        j = min(max(j, 0), len(t) - 1)
        if j > 0 and abs(t[j - 1] - e["time_s"]) < abs(t[j] - e["time_s"]):
            j -= 1
        cx, cy = td.col(e["team"], e["from_player"], "x"), td.col(e["team"], e["from_player"], "y")
        if cx not in f.columns:
            out.append((e.name, np.nan, np.nan, False, np.nan)); continue
        px, py = f[cx].to_numpy(float), f[cy].to_numpy(float)
        ks = np.arange(-PRE_WINDOW, POST_FRAMES + 1)
        idx = j + ks
        ok = (idx >= 0) & (idx < len(t))
        ok[ok] = (per[idx[ok]] == per[j]) & (fr[idx[ok]] - fr[j] == ks[ok])
        d = np.where(ok, np.hypot(bx[np.clip(idx, 0, len(t) - 1)] - px[np.clip(idx, 0, len(t) - 1)], by[np.clip(idx, 0, len(t) - 1)] - py[np.clip(idx, 0, len(t) - 1)]), np.nan)
        d0 = d[PRE_WINDOW]
        pre = d[: PRE_WINDOW + 1]
        censored = bool(pre[0] > RELEASE_DIST_M) if np.isfinite(pre[0]) else False
        cross = np.flatnonzero(d > RELEASE_DIST_M)
        tau = ks[cross[0]] * 0.04 if len(cross) else np.nan
        post = d[PRE_WINDOW:]
        # ball travel over the 6 frames after the event frame
        i0, i1 = j, min(j + POST_FRAMES, len(t) - 1)
        speed = np.hypot(bx[i1] - bx[i0], by[i1] - by[i0]) / (t[i1] - t[i0]) if i1 > i0 and fr[i1] - fr[i0] == i1 - i0 else np.nan
        out.append((e.name, d0, tau, censored, speed))
    return pd.DataFrame(out, columns=["row", "d0", "tau_s", "censored", "speed_after"]).set_index("row")


def lag_profile(td, ev: pd.DataFrame, lags=range(-10, 11, 2)) -> pd.DataFrame:
    """Median ball-passer distance and median ball speed at event frame + lag (frames), for visual inspection of alignment."""
    f = td.frames
    t = f["time_s"].to_numpy(float)
    bx, by = f["ball_x"].to_numpy(float), f["ball_y"].to_numpy(float)
    rows = {k: ([], []) for k in lags}
    for _, e in ev.iterrows():
        cx, cy = td.col(e["team"], e["from_player"], "x"), td.col(e["team"], e["from_player"], "y")
        if cx not in f.columns:
            continue
        j = min(max(int(np.searchsorted(t, e["time_s"])), 1), len(t) - 2)
        px, py = f[cx].to_numpy(float), f[cy].to_numpy(float)
        for k in lags:
            i = j + k
            if 1 <= i < len(t):
                rows[k][0].append(np.hypot(bx[i] - px[i], by[i] - py[i]))
                rows[k][1].append(np.hypot(bx[i] - bx[i - 1], by[i] - by[i - 1]) / (t[i] - t[i - 1]) if t[i] > t[i - 1] else np.nan)
    return pd.DataFrame({"lag_frames": list(lags), "lag_s": [k * 0.04 for k in lags], "median_dist_m": [np.nanmedian(rows[k][0]) for k in lags],
                         "median_ball_speed_ms": [np.nanmedian(rows[k][1]) for k in lags]}).round(2)


def metrica_reference_events():
    """Metrica Game 1 passes with the SAME population rule as the existing pipeline (read-only)."""
    from src.cli import _load
    from src.config import load_config
    from src.pass_completion_features import label_events
    cfg = load_config()
    td, ev = _load(cfg, 1)
    lab = label_events(ev)
    cand = ev.loc[lab[lab.outcome.notna()].event_idx]
    sync = synchronize_events(cand, td.frames, cfg["sync"]["tolerance_s"], "start_time_s", None).matched
    e = cand.loc[sync.event_idx].assign(time_s=lambda d: d["start_time_s"])
    return td, e[["time_s", "team", "from_player"]]


def summarize_release(m: pd.DataFrame) -> dict:
    nc = m[~m.censored]
    return {"n": len(m), "median_d0": float(m.d0.median()), "share_d0_over_3m": float((m.d0 > 3).mean()), "tau_median": float(nc.tau_s.median()),
            "tau_q25": float(nc.tau_s.quantile(.25)), "tau_q75": float(nc.tau_s.quantile(.75)), "censored": float(m.censored.mean()),
            "speed_after_median": float(m.speed_after.median()), "tau_never_crosses": float(m.tau_s.isna().mean())}


def gate_checks(a: dict, b: dict) -> dict[str, bool]:
    """The five pre-registered criteria. ``a`` = candidate source summary, ``b`` = Metrica reference summary."""
    iqr_a, iqr_b = a["tau_q75"] - a["tau_q25"], b["tau_q75"] - b["tau_q25"]
    return {
        "median ball-passer distance at event frame <= 0.5 m": a["median_d0"] <= GATE["median_d0_max_m"],
        "share with ball > 3 m from the passer <= 3%": a["share_d0_over_3m"] <= GATE["share_d0_over_3m_max"],
        "median release lag within 0.08 s of Metrica": bool(abs(a["tau_median"] - b["tau_median"]) <= GATE["median_tau_diff_max_s"]),
        "release-lag IQR <= 2x Metrica": bool(iqr_a <= GATE["iqr_ratio_max"] * iqr_b),
        "median ball speed after event >= 8 m/s": a["speed_after_median"] >= GATE["median_speed_after_min_ms"],
    }


def gate_release(td_i, ev_i: pd.DataFrame, say) -> bool:
    ref_td, ref_ev = metrica_reference_events()
    res = {}
    say(f"[release gate] criteria fixed beforehand: {GATE}")
    for nm, (td, e) in (("IDSSE J03WPY", (td_i, ev_i)), ("Metrica G1 (reference)", (ref_td, ref_ev))):
        res[nm] = summarize_release(release_measure(td, e))
        say(f"  {nm}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res[nm].items()))
    checks = gate_checks(res["IDSSE J03WPY"], res["Metrica G1 (reference)"])
    for k, v in checks.items():
        say(f"  [{'PASS' if v else 'FAIL'}] {k}")
    say("[lag profile] IDSSE (event frame + lag):\n" + lag_profile(td_i, ev_i).to_string(index=False))
    say("[lag profile] Metrica G1 reference:\n" + lag_profile(ref_td, ref_ev).to_string(index=False))
    return all(checks.values())


def diagnose_release(td, ev_pass: pd.DataFrame, say) -> None:
    """DIAGNOSTIC ONLY, run after a gate failure: is the problem a constant clock offset or imprecise event times? Nothing here
    re-times events or changes any definition."""
    f = td.frames
    t, bx, by = f["time_s"].to_numpy(float), f["ball_x"].to_numpy(float), f["ball_y"].to_numpy(float)
    best, d0 = [], []
    for _, e in ev_pass.iterrows():
        cx, cy = td.col(e["team"], e["from_player"], "x"), td.col(e["team"], e["from_player"], "y")
        if cx not in f.columns:
            continue
        j = int(np.searchsorted(t, e["time_s"]))
        ks = np.arange(-125, 126)
        idx = np.clip(j + ks, 0, len(t) - 1)
        d = np.hypot(bx[idx] - f[cx].to_numpy(float)[idx], by[idx] - f[cy].to_numpy(float)[idx])
        best.append((ks[np.nanargmin(d)] * 0.04, np.nanmin(d)))
    b = pd.DataFrame(best, columns=["lag_s", "dmin"])
    say(f"[diagnostic] per-event lag at which the ball is closest to the named passer (+-5 s): median {b.lag_s.median():+.2f} s, IQR {b.lag_s.quantile(.25):+.2f}..{b.lag_s.quantile(.75):+.2f} s; "
        f"share within +-0.12 s (3 frames): {(b.lag_s.abs() <= 0.12).mean():.1%}; median minimum distance {b.dmin.median():.2f} m")
    say("[diagnostic] ball-passer distance vs lag (median [IQR]) over +-4 s:")
    for k in range(-100, 101, 25):
        ds = []
        for _, e in ev_pass.iterrows():
            cx, cy = td.col(e["team"], e["from_player"], "x"), td.col(e["team"], e["from_player"], "y")
            j = int(np.searchsorted(t, e["time_s"])) + k
            if cx in f.columns and 0 <= j < len(t):
                ds.append(np.hypot(bx[j] - f[cx].to_numpy(float)[j], by[j] - f[cy].to_numpy(float)[j]))
        say(f"    lag {k * 0.04:+.1f} s: {np.nanmedian(ds):.2f} m [{np.nanpercentile(ds, 25):.2f}..{np.nanpercentile(ds, 75):.2f}]")


def load_all():
    meta = dfl.read_match_info(DIR / FILES["meta"])
    td, qa, t0 = dfl.read_tracking(DIR / FILES["positions"], DIR / FILES["meta"])
    f = td.frames
    bounds = {int(p): (float(g.time_s.min()), float(g.time_s.max())) for p, g in f.groupby("period")}
    ev = dfl.read_events(DIR / FILES["events"], meta, t0, bounds)
    return meta, td, qa, ev


def quality(meta, td, qa, ev, say) -> None:
    f = td.frames
    say(f"[tracking] frames {len(f)}, objects/framesets {qa.n_framesets}, tracked persons {len(td.players)} (metadata lists {len(meta.players)}; never tracked {len(qa.metadata_players_never_tracked)}; tracked but not in metadata {qa.persons_not_in_metadata})")
    dt = f.groupby("period")["time_s"].diff().dropna()
    say(f"[frequency] frame spacing: median {dt.median():.3f} s -> {1 / dt.median():.2f} Hz; spacing != 0.040 s in {(abs(dt - 0.04) > 1e-6).sum()} steps; duplicate (period, frame) {f.duplicated(['period', 'frame']).sum()}; "
        f"frame-number gaps {(f.groupby('period')['frame'].diff().dropna() != 1).sum()}; T disagreement across objects {qa.t_mismatch_across_objects}")
    say("[periods] " + "; ".join(f"P{p}: {g.time_s.min():.1f}-{g.time_s.max():.1f} s, {len(g)} frames (frames {g.frame.min()}-{g.frame.max()})" for p, g in f.groupby("period")))
    say(f"[missing] NaN coordinates parsed as NaN: {qa.missing_xy}; frames without a ball record {qa.frames_without_ball}; ball status: {f.ball_status.value_counts().to_dict()}")
    pcols = td.player_columns()
    xs = f[[c for c in pcols if c.endswith('__x')]]
    on = xs.notna().sum(axis=1)
    say(f"[players on pitch per frame] min {on.min()}, median {int(on.median())}, max {on.max()}; frames with fewer than 22 tracked: {(on < 22).mean():.2%}")
    allx = np.concatenate([f[[c for c in pcols if c.endswith('__x')]].to_numpy(float).ravel(), f['ball_x'].to_numpy(float)])
    ally = np.concatenate([f[[c for c in pcols if c.endswith('__y')]].to_numpy(float).ravel(), f['ball_y'].to_numpy(float)])
    say(f"[coordinates] pitch {meta.pitch_x} x {meta.pitch_y} (metadata) = {qa.pitch_size_tracking} (positions); x range {np.nanmin(allx):.1f}..{np.nanmax(allx):.1f}, y range {np.nanmin(ally):.1f}..{np.nanmax(ally):.1f} "
        f"(centre origin, metres); beyond pitch+3 m: {int(((abs(allx) > meta.pitch_x / 2 + 3) | (abs(ally) > meta.pitch_y / 2 + 3)).sum())} values")
    rep = validate_tracking(td, {"validation": {}})
    say("[existing validation report]\n" + rep.to_text())
    # attack direction vs the KickOff TeamLeft attribute
    ko = ev[ev["chain"].str.startswith("KickOff") & ev["raw"].str.contains("firstHalf")]
    import json
    left = json.loads(ko.iloc[0]["raw"])["KickOff.TeamLeft"]
    left_side = meta.team_side[left]
    say(f"[orientation] KickOff TeamLeft = {left} ({left_side}); inferred attack_dir {td.attack_dir}; consistent: {td.attack_dir[(left_side, 1)] == 1}")
    # identities
    team_of = dict(zip(td.players.player_id, td.players.team))
    play = ev[ev.is_play & ~ev.is_delete]
    unknown_from = int((~play.from_player.isin(team_of)).sum())
    unknown_rec = int((play.recipient.notna() & ~play.recipient.isin(team_of)).sum())
    wrong_team = int(sum(team_of.get(p) not in (None, t) for p, t in zip(play.from_player, play.team)))
    say(f"[identities] Play events {len(play)}; passer not tracked {unknown_from}; recipient not tracked {unknown_rec}; passer on the wrong team {wrong_team}; recipient missing {int(play.recipient.isna().sum())}")
    swaps = detect_identity_swaps(td, play.assign(start_time_s=play.time_s)[["team", "period", "start_time_s", "from_player"]].dropna(subset=["period"]).rename_axis(None))
    say(f"[identity-swap detector] {swaps if swaps else 'none detected'}")
    # events vs tracking clock
    ev_p = ev[ev.is_play & ~ev.is_delete & ev.period.notna()]
    res = synchronize_events(ev_p.assign(start_time_s=ev_p.time_s)[["team", "period", "start_time_s", "from_player"]], td.frames, 0.02, "start_time_s", None)
    say(f"[event sync] Play events matched to a frame within 0.02 s: {len(res.matched)}/{len(ev_p)}; unmatched reasons {res.unmatched.reason.str.split(' \\(').str[0].value_counts().to_dict() if len(res.unmatched) else {}}; events outside both periods: {int(ev[ev.is_play].period.isna().sum())}")
    # StartFrame / CalculatedFrame cross-checks (provider-supplied frame references)
    import json as js
    chk = []
    for _, e in ev.iterrows():
        r = js.loads(e.raw)
        for key in ("StartFrame", "CalculatedFrame"):
            if key in r:
                chk.append((e.event_id, key, int(r[key]), e.time_s))
    if chk:
        fr = td.frames.set_index("frame")["time_s"]
        diffs = [(k, e_t - float(fr.get(n, np.nan))) for _, k, n, e_t in chk]
        for key in ("StartFrame", "CalculatedFrame"):
            v = [d for k, d in diffs if k == key and np.isfinite(d)]
            if v:
                say(f"[provider frame refs] {key}: n={len(v)}, EventTime - tracking time of that frame: median {np.median(v):+.3f} s, range {min(v):+.3f}..{max(v):+.3f} s")
    # substitutions vs tracking windows
    subs = ev[ev.chain.str.startswith("Substitution")]
    lines = []
    for _, s in subs.iterrows():
        r = js.loads(s.raw)
        out_p, in_p = r.get("Substitution.PlayerOut"), r.get("Substitution.PlayerIn")
        for nm, pid in (("out", out_p), ("in", in_p)):
            cx = None
            for t in ("Home", "Away"):
                c = td.col(t, pid, "x")
                if c in td.frames.columns:
                    cx = c
            if cx is None:
                lines.append(f"{nm} {pid}: not tracked"); continue
            ser = td.frames.loc[td.frames[cx].notna(), "time_s"]
            lines.append(f"{nm} {pid}: sub at {s.time_s:.0f}s, tracked {ser.min():.0f}-{ser.max():.0f}s")
    say("[substitutions] " + "; ".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage", choices=["quality", "gate", "labels", "features"], default="features")
    a = ap.parse_args(argv)
    L = []
    say = L.append

    def done(code):
        text = "\n".join(L)
        print(text)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text + "\n")
        return code

    meta, td, qa, ev = load_all()
    quality(meta, td, qa, ev, say)
    if a.stage == "quality":
        return done(0)
    elig = frozen_eligibility(ev)
    passes = elig[elig.is_play & ~elig.is_delete]
    say(f"\n[frozen eligibility] Play events {len(passes)}: eligible {int(passes.eligible.sum())}; excluded: {passes[~passes.eligible].excluded_reason.value_counts().to_dict()}")
    say("[headers] NOT excludable: IDSSE passes carry no header flag (see idsse_labels.py); header passes remain in the eligible set")
    gate_ev = passes[passes.eligible & passes.period.notna()].assign(time_s=lambda d: d.time_s)[["time_s", "team", "from_player"]]
    say("\n== CRITICAL GATE: does EventTime identify the pass release moment? ==")
    if not gate_release(td, gate_ev, say):
        say("\nGATE FAILED -> STOP. No labels or features were computed.")
        diagnose_release(td, passes[passes.eligible & passes.period.notna()], say)
        return done(1)
    say("\nRELEASE-FRAME GATE PASSED")
    if a.stage == "gate":
        return done(0)
    person_team = dict(zip(td.players.player_id, td.players.team))
    la, lb = label_a(elig, person_team), label_b(elig, person_team)
    c = compare(la, lb)
    say("\n== LABEL AGREEMENT (A: opposing Recipient; B: following opposing BallClaiming InterceptedBall) ==")
    say(str(c))
    say(f"reasons A: {la[la.outcome.isna()].reason.value_counts().to_dict()}")
    say(f"reasons B: {lb[lb.outcome.isna()].reason.value_counts().to_dict()}")
    if not (c["jaccard"] >= JACCARD_MIN):
        say(f"\nMATERIAL DISAGREEMENT (Jaccard {c['jaccard']:.2f} < {JACCARD_MIN}, fixed beforehand) -> STOP. No features were computed.")
        return done(1)
    say("\nLABELS AGREE")
    return done(0)


if __name__ == "__main__":
    raise SystemExit(main())
