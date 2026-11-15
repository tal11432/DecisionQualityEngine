"""Goal-by-goal review of one match with the frozen DQ models (xG_geo + continuation for the shot; xP x chain value for the best visible teammate). Usage: python research/cases/match_review.py MATCH_ID"""
import sys, json, gzip, urllib.request, numpy as np, pandas as pd
sys.path.insert(0, "research/dq"); from dscore.core import Situation, shot_value, pass_options, rank_options, verdict as vv, GX, SC     # helpers shared with the Decision Score (research/dq/dscore/core.py)
mid = sys.argv[1]
B = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
def get(p):
    r = urllib.request.urlopen(urllib.request.Request(B + p, headers={"Accept-Encoding": "gzip"}), timeout=60); b = r.read(); return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
def secs(e): h, m, s = e["timestamp"].split(":"); return e["period"] * 10000 + int(h) * 3600 + int(m) * 60 + float(s)
ev = get(f"events/{mid}.json"); byid = {e["id"]: e for e in ev}
sp = [(secs(e), e["team"]["id"], e["possession"], e["shot"]["statsbomb_xg"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play"]
teams = sorted({e["team"]["name"] for e in ev if "team" in e})[:2]
goals = [e for e in ev if (e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal") or e["type"]["name"] == "Own Goal For"]
summary = []
print(f"MATCH {mid}: {teams[0]} vs {teams[1]}; goals found: {len(goals)}\n")
for e in goals:
    mn = f"{e['minute']}:{e['second']:02d}"; tm = e["team"]["name"]
    if e["type"]["name"] == "Own Goal For": print(f"== {mn}  {tm}: own goal by the opponent. Not a decision of the scoring side, no verdict.\n"); continue
    s = e["shot"]; kind = s["type"]["name"]; body = s["body_part"]["name"]; scorer = e["player"]["name"]; kp = byid.get(s.get("key_pass_id")); kpp = kp["pass"] if kp else {}
    how = ("cross" if kpp.get("cross") else "through ball" if kpp.get("technique", {}).get("name") == "Through Ball" else "cut-back" if kpp.get("cut_back") else "ground pass" if kp else "no key pass (own action)")
    assist = f" (assist: {kp['player']['name']}, {how})" if kp else ""
    line = f"== {mn}  {tm}: {scorer}, {body.lower()}, {kind.lower()}, play pattern {e['play_pattern']['name']}{assist}; StatsBomb xG {s['statsbomb_xg']:.2f}"
    print(line)
    if kind != "Open Play": print(f"   Not an open-play decision ({kind.lower()}): no alternative to compare.\n"); continue
    fr = s.get("freeze_frame", []); a = np.array(e["location"][:2], float); O = np.array([q["location"][:2] for q in fr if not q["teammate"]], float).reshape(-1, 2); Tm = [np.array(q["location"][:2], float) for q in fr if q["teammate"]]
    gks = [q["location"][:2] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
    if not gks: print("   no goalkeeper in the frame: cannot evaluate.\n"); continue
    sit = Situation(a, O, Tm, gks[0]); hd = int(body == "Head"); xg, cont, shot_val, f2 = shot_value(sit, hd); dist = f2["dist"]
    live, sure, gone = rank_options(pass_options(sit))
    print(f"   Shooter {dist*105/120:.1f} m from goal; defenders in the shooting triangle (keeper included): {f2['n_tri']}; visible teammates {len(Tm)}; options: {len(sure)} certain onside, {len(live)-len(sure)} in doubt, {len(gone)} certainly offside; visible opponents {len(O)} of 11")
    print(f"   Model: shot value {shot_val:.2f} (xG {xg:.2f} + continuation {cont:.2f})")
    if hd: print("   This was a header: a pass was not a realistic alternative, so the comparison is shown but should not be read as advice.")
    elif kpp.get("cross"): print("   This was a first-time finish from a cross: passing was rarely a realistic alternative, so read the comparison with caution.")
    def verdict(o): return vv(o['v'], shot_val, o['d'])
    for c in live[:3]: print(f"     option: {c['d']:.0f} m away, {c['g']:.0f} m from goal; reaches him {c['xp']*100:.0f}%, chain value {c['xa']:.2f} => pass value {c['v']:.2f}; offside margin {c['margin']:+.1f} m => {c['status']}")
    rec = dict(minute=mn, team=tm, scorer=scorer, how=("header" if hd else ("first-time from a cross" if kpp.get("cross") else "foot")), shot=shot_val, dist=dist * 105 / 120, best=live[0] if live else None, sure=sure[0] if sure else None)
    if not live: print("   No receiver in view that is onside or in doubt. VERDICT: the shot was the only visible option.\n"); summary.append(rec); continue
    b, c0 = live[0], (sure[0] if sure else None)
    if b["status"] != "certain onside":
        print(f"   The best-valued option is in DOUBT; the next option that is certainly onside: " + (f"{c0['d']:.0f} m away, {c0['g']:.0f} m from goal, value {c0['v']:.2f}, onside by {c0['margin']:.1f} m" if c0 else "none in view"))
        print(f"   VERDICT: counting the doubtful option: {verdict(b)} (pass {b['v']:.2f} vs shot {shot_val:.2f}); counting only certain options: " + (f"{verdict(c0)} (pass {c0['v']:.2f} vs shot {shot_val:.2f})" if c0 else "the shot was the only certain option") + "\n")
    else: print(f"   VERDICT: {verdict(b)}. Best pass value {b['v']:.2f} (certainly onside by {b['margin']:.1f} m) vs shot value {shot_val:.2f} (ratio {b['v']/shot_val:.2f}).\n")
    summary.append(rec)
print("\nSUMMARY TABLE")
print("| minute | team / scorer | kind | shot value | best option (value, status) | next certainly-onside option | verdict (certain options only) | verdict if the doubtful option counts |")
print("|---|---|---|---|---|---|---|---|")

for r in summary:
    b, c0 = r["best"], r["sure"]
    if b is None: print(f"| {r['minute']} | {r['team']} / {r['scorer']} | {r['how']} | {r['shot']:.2f} | none in view | none | shot only option | shot only option |"); continue
    st = "certain onside" if b["status"] == "certain onside" else "DOUBT"
    nxt = "(same)" if b["status"] == "certain onside" else (f"{c0['v']:.2f} ({c0['d']:.0f} m away)" if c0 else "none")
    vc = vv(c0["v"], r["shot"], c0["d"]) if c0 else "shot only certain option"; va = vv(b["v"], r["shot"], b["d"])
    print(f"| {r['minute']} | {r['team']} / {r['scorer']} | {r['how']} | {r['shot']:.2f} | {b['v']:.2f} ({st}, {b['margin']:+.1f} m) | {nxt} | {vc} | {va} |")
