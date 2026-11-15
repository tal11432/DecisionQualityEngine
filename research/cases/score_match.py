"""Decision Score for every shot, ground pass and carry at x >= 80 of one match (StatsBomb open data; 360 frames needed for passes and carries). Usage: python research/cases/score_match.py MATCH_ID
No player names except the scorers of goals; verdicts use the noise floors of DQ-v12b (a gap below the floor is 'no clear difference', however many points it looks like)."""
import sys, json, gzip, urllib.request, numpy as np, pandas as pd
sys.path.insert(0, "research/dq"); from dscore import score as S; from dscore.core import Situation, SC
mid = sys.argv[1]; B = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
def get(p):
    try:
        r = urllib.request.urlopen(urllib.request.Request(B + p, headers={"Accept-Encoding": "gzip"}), timeout=60); b = r.read(); return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
    except urllib.error.HTTPError: return None
def secs(e): h, m, s = e["timestamp"].split(":"); return e["period"] * 10000 + int(h) * 3600 + int(m) * 60 + float(s)
ev = get(f"events/{mid}.json"); r = get(f"three-sixty/{mid}.json"); ff = {f["event_uuid"]: f for f in r} if r else {}
shots = [(secs(e), e["team"]["id"], e["shot"]["statsbomb_xg"], e["possession"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play"]; n = len(ev); H = 10.0
dec = []
for i, e in enumerate(ev):
    t = e["type"]["name"]; loc = e.get("location")
    if loc is None or loc[0] < 80: continue
    a = np.array(loc[:2], float); t0 = secs(e); tm = e["team"]["id"]; row = None
    if t == "Shot" and e["shot"]["type"]["name"] == "Open Play" and e["shot"]["body_part"]["name"] != "Head":
        fr = e["shot"].get("freeze_frame", []); O = [q["location"][:2] for q in fr if not q["teammate"]]; Tm = [q["location"][:2] for q in fr if q["teammate"]]; gks = [q["location"][:2] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
        if gks and O: row = dict(kind="shot", opps=O, mates=Tm, gk=gks[0], goal=int(e["shot"]["outcome"]["name"] == "Goal"), scorer=e["player"]["name"] if e["shot"]["outcome"]["name"] == "Goal" else "")
    elif e["id"] in ff and t in ("Pass", "Carry"):
        fr = ff[e["id"]]["freeze_frame"]; O = [q["location"][:2] for q in fr if not q["teammate"]]; Tm = [q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")]; kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]
        if kp_ and O:
            if t == "Pass":
                ps = e["pass"]
                if not ("type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" or ps.get("body_part", {}).get("name") == "Head" or "end_location" not in ps):
                    row = dict(kind="pass", tx=ps["end_location"][0], ty=ps["end_location"][1], completed=int("outcome" not in ps), opps=O, mates=Tm, gk=kp_[0])
            elif "end_location" in e["carry"] and float(np.hypot(*(np.array(e["carry"]["end_location"][:2]) - a))) >= 1.0: row = dict(kind="carry", tx=e["carry"]["end_location"][0], ty=e["carry"]["end_location"][1], opps=O, mates=Tm, gk=kp_[0])
    if row is None: continue
    row.update(team=e["team"]["name"], clock=f"{e['minute']:02d}:{e['second']:02d}", ax=a[0], ay=a[1], own10=sum(x for tt, ti, x, pp in shots if ti == tm and t0 < tt <= t0 + H), opp10=sum(x for tt, ti, x, pp in shots if ti != tm and t0 < tt <= t0 + H)); dec.append(row)
rows = []
for d in dec:
    sit = Situation([d["ax"], d["ay"]], np.array(d["opps"], float).reshape(-1, 2), d["mates"], d["gk"]); ch = ("shot",) if d["kind"] == "shot" else (d["kind"], [d["tx"], d["ty"]]); j = S.judge(sit, ch); jd = S.judge(sit, ch, include_doubt=True)
    floor = 0.05 if (j["chosen"].get("d", 0) > 15 and d["kind"] == "pass") or (j["best"]["kind"] == "pass" and j["best"].get("d", 0) > 15) else 0.03
    verdict = "no clear difference from the best option" if j["gap_xg"] < floor else "clearly below the best option"
    rows.append(dict(clock=d["clock"], team=d["team"], kind=d["kind"] + ("" if d["kind"] != "pass" else (" (completed)" if d["completed"] else " (failed)")), score=j["score_chosen"], best_kind=j["best"]["kind"], best_score=j["score_best"], gap_points=j["gap_points"], gap_xg=j["gap_xg"], q=j["q_chosen"], q_best=j["q_best"], gap_with_doubt_xg=jd["gap_xg"],
                     options_certain=sum(o["status"] == "certain onside" for o in j["options"] if o["kind"] == "pass"), verdict=verdict, goal=d.get("goal", 0), scorer=d.get("scorer", ""), realised=(np.nan if d["kind"] == "shot" else d["own10"] - d["opp10"])))
R = pd.DataFrame(rows); teams = sorted(R.team.unique()); print(f"MATCH {mid}: {len(R)} scored decisions at x >= 80 ({dict(R.kind.str.split(' ').str[0].value_counts())}); frames for passes/carries: {'yes' if ff else 'NO (shots only)'}")
print(f"score of the option taken: median {R.score.median():.0f}, 10th/90th {R.score.quantile(.1):.0f}/{R.score.quantile(.9):.0f}; gap >= the noise floor in {(R.verdict.str.startswith('clearly')).mean()*100:.0f}% of decisions (the rest: no clear difference from the best option)")
for tm, g in R.groupby("team"): print(f"   {tm}: {len(g)} decisions, median score {g.score.median():.0f}, mean Q {g.q.mean():+.3f}, clearly-below-best {(g.verdict.str.startswith('clearly')).mean()*100:.0f}%")
print("\nGOALS (shots that scored):"); print(R[R.goal == 1][["clock", "team", "scorer", "score", "best_kind", "best_score", "gap_points", "gap_xg", "verdict"]].round(3).to_string(index=False))
print("\nLARGEST CLEAR GAPS (the option taken was clearly below the best one):"); print(R[R.verdict.str.startswith("clearly")].sort_values("gap_xg", ascending=False).head(8)[["clock", "team", "kind", "score", "best_kind", "best_score", "gap_xg", "q", "q_best"]].round(3).to_string(index=False))
print("\nHIGHEST SCORES:"); print(R.sort_values("score", ascending=False).head(6)[["clock", "team", "kind", "score", "gap_xg", "q", "realised"]].round(3).to_string(index=False))
R.drop(columns=["scorer"]).round(4).to_csv(f"research/cases/decision_scores_{mid}.csv", index=False); print(f"\nwrote research/cases/decision_scores_{mid}.csv (no player names)")
