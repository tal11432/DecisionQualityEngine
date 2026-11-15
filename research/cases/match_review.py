"""Goal-by-goal review of one match with the frozen DQ models (xG_geo + continuation for the shot; xP x chain value for the best visible teammate). Usage: python research/cases/match_review.py MATCH_ID"""
import sys, json, gzip, urllib.request, numpy as np, pandas as pd
sys.path.insert(0, "research/dq"); from tfcommon import steps, FEATS as HFT
GX = 120.0; SC = np.array([105 / 120, 68 / 80]); mid = sys.argv[1]
B = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
def get(p):
    r = urllib.request.urlopen(urllib.request.Request(B + p, headers={"Accept-Encoding": "gzip"}), timeout=60); b = r.read(); return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
M5 = pd.read_pickle("research/dq/v5_models.pkl"); M6 = pd.read_pickle("research/dq/v6_models.pkl"); HM = pd.read_pickle("research/dq/tf_model.pkl")
sgn = lambda m, X: float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))
ang = lambda x, y: float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def tri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def hz(a, end, O):
    R = steps(a, end, O, False, None); l = np.column_stack([np.ones(len(R)), (R[:, :len(HFT)] - HM["mu"]) / HM["sd"]]) @ HM["b"]; return float(np.exp(np.log1p(-1 / (1 + np.exp(-l))).sum()))
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
    gk = np.array(gks[0], float); dd = np.hypot(*(O - a).T); v = np.array([GX, 40.0]) - a; dist = float(np.hypot(*v)); an = ang(*a); gl = float(abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9))
    hd = int(body == "Head"); f2 = dict(dist=dist, angle=an, header=hd, n_tri=sum(tri(o, a) for o in O), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), gk_lat=gl, n_opp2=int((dd <= 2).sum()), nn_opp=float(dd.min()), da=dist * an, ldist=float(np.log(dist)))
    xg = sgn(M5["xG_geo"], [f2[k] for k in M5["xG_geo"]["F"]]); cont = sgn(M6["cont"], [f2[k] for k in M6["cont"]["F"]]); shot_val = xg + cont
    t0 = secs(e); ox = np.sort(O[:, 0]); second_last = ox[-2] if len(ox) > 1 else ox[-1]; lineo = max(second_last, a[0]); cands = []
    for p in Tm:
        d = float(np.hypot(*((p - a) * SC)))
        if not (3 <= d <= 40): continue
        margin_m = float((lineo - p[0]) * 105 / 120)          # > 0: onside by this many metres; < 0: offside by this many metres (position proxy)
        if p[0] <= 60 or margin_m >= 1.0: status = "certain onside"
        elif margin_m > -1.0: status = "DOUBT (within 1 m of the line)"
        elif len(O) < 11: status = "DOUBT (offside by the visible players, but a defender is out of view)"
        else: status = "certain offside"
        dp = np.hypot(*(O - p).T); dist_p = float(np.hypot(GX - p[0], 40 - p[1])); an_p = ang(*p)
        ft = dict(dist=dist_p, angle=an_p, x=p[0], d2=dist_p ** 2, ad=an_p * dist_p, n_tri=sum(tri(o, p) for o in O), n3=int((dp <= 3).sum()), n6=int((dp <= 6).sum()), nn=float(dp.min()), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), plen=float(np.hypot(*(p - a))), fwd=float(p[0] - a[0]), pdist=dist)
        xa = sgn(M6["xA_chain"], [ft[k] for k in M6["xA_chain"]["F"]]); xp = hz(a, p, O); cands.append(dict(v=xp * xa, xp=xp, xa=xa, d=d, g=dist_p * 105 / 120, margin=margin_m, status=status))
    live = sorted([c for c in cands if c["status"] != "certain offside"], key=lambda c: -c["v"]); sure = [c for c in live if c["status"] == "certain onside"]; gone = [c for c in cands if c["status"] == "certain offside"]
    print(f"   Shooter {dist*105/120:.1f} m from goal; defenders in the shooting triangle (keeper included): {f2['n_tri']}; visible teammates {len(Tm)}; options: {len(sure)} certain onside, {len(live)-len(sure)} in doubt, {len(gone)} certainly offside; visible opponents {len(O)} of 11")
    print(f"   Model: shot value {shot_val:.2f} (xG {xg:.2f} + continuation {cont:.2f})")
    if hd: print("   This was a header: a pass was not a realistic alternative, so the comparison is shown but should not be read as advice.")
    elif kpp.get("cross"): print("   This was a first-time finish from a cross: passing was rarely a realistic alternative, so read the comparison with caution.")
    def verdict(bv): return "pass clearly better" if 0.7 * bv > shot_val else ("shot clearly better" if 1.3 * bv < shot_val else "ambiguous")
    for c in live[:3]: print(f"     option: {c['d']:.0f} m away, {c['g']:.0f} m from goal; reaches him {c['xp']*100:.0f}%, chain value {c['xa']:.2f} => pass value {c['v']:.2f}; offside margin {c['margin']:+.1f} m => {c['status']}")
    rec = dict(minute=mn, team=tm, scorer=scorer, how=("header" if hd else ("first-time from a cross" if kpp.get("cross") else "foot")), shot=shot_val, dist=dist * 105 / 120, best=live[0] if live else None, sure=sure[0] if sure else None)
    if not live: print("   No receiver in view that is onside or in doubt. VERDICT: the shot was the only visible option.\n"); summary.append(rec); continue
    b, c0 = live[0], (sure[0] if sure else None)
    if b["status"] != "certain onside":
        print(f"   The best-valued option is in DOUBT; the next option that is certainly onside: " + (f"{c0['d']:.0f} m away, {c0['g']:.0f} m from goal, value {c0['v']:.2f}, onside by {c0['margin']:.1f} m" if c0 else "none in view"))
        print(f"   VERDICT: counting the doubtful option: {verdict(b['v'])} (pass {b['v']:.2f} vs shot {shot_val:.2f}); counting only certain options: " + (f"{verdict(c0['v'])} (pass {c0['v']:.2f} vs shot {shot_val:.2f})" if c0 else "the shot was the only certain option") + "\n")
    else: print(f"   VERDICT: {verdict(b['v'])}. Best pass value {b['v']:.2f} (certainly onside by {b['margin']:.1f} m) vs shot value {shot_val:.2f} (ratio {b['v']/shot_val:.2f}).\n")
    summary.append(rec)
print("\nSUMMARY TABLE")
print("| minute | team / scorer | kind | shot value | best option (value, status) | next certainly-onside option | verdict (certain options only) | verdict if the doubtful option counts |")
print("|---|---|---|---|---|---|---|---|")
vv = lambda bv, sv: "pass clearly better" if 0.7 * bv > sv else ("shot clearly better" if 1.3 * bv < sv else "ambiguous")
for r in summary:
    b, c0 = r["best"], r["sure"]
    if b is None: print(f"| {r['minute']} | {r['team']} / {r['scorer']} | {r['how']} | {r['shot']:.2f} | none in view | none | shot only option | shot only option |"); continue
    st = "certain onside" if b["status"] == "certain onside" else "DOUBT"
    nxt = "(same)" if b["status"] == "certain onside" else (f"{c0['v']:.2f} ({c0['d']:.0f} m away)" if c0 else "none")
    vc = vv(c0["v"], r["shot"]) if c0 else "shot only certain option"; va = vv(b["v"], r["shot"])
    print(f"| {r['minute']} | {r['team']} / {r['scorer']} | {r['how']} | {r['shot']:.2f} | {b['v']:.2f} ({st}, {b['margin']:+.1f} m) | {nxt} | {vc} | {va} |")
