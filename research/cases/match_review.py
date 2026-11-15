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
    t0 = secs(e); ox = np.sort(O[:, 0]); lineo = max(ox[-2] if len(ox) > 1 else ox[-1], a[0]); best = (0.0, None); cands = []
    for p in Tm:
        d = float(np.hypot(*((p - a) * SC)))
        if not (3 <= d <= 40) or (p[0] > lineo and p[0] > 60): continue
        dp = np.hypot(*(O - p).T); dist_p = float(np.hypot(GX - p[0], 40 - p[1])); an_p = ang(*p)
        ft = dict(dist=dist_p, angle=an_p, x=p[0], d2=dist_p ** 2, ad=an_p * dist_p, n_tri=sum(tri(o, p) for o in O), n3=int((dp <= 3).sum()), n6=int((dp <= 6).sum()), nn=float(dp.min()), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), plen=float(np.hypot(*(p - a))), fwd=float(p[0] - a[0]), pdist=dist)
        xa = sgn(M6["xA_chain"], [ft[k] for k in M6["xA_chain"]["F"]]); xp = hz(a, p, O); cands.append((xp * xa, xp, xa, d, dist_p * 105 / 120)); best = max(best, (xp * xa, p), key=lambda z: z[0])
    cands.sort(reverse=True)
    print(f"   Shooter {dist*105/120:.1f} m from goal; defenders in the shooting triangle (keeper included): {f2['n_tri']}; visible teammates: {len(Tm)}, admissible receivers: {len(cands)}")
    print(f"   Model: shot value {shot_val:.2f} (xG {xg:.2f} + continuation {cont:.2f})")
    if hd: print("   This was a header: a pass was not a realistic alternative, so the comparison is shown but should not be read as advice.")
    elif kpp.get("cross"): print("   This was a first-time finish from a cross: passing was rarely a realistic alternative, so read the comparison with caution.")
    if cands:
        for c in cands[:3]: print(f"     best pass option: {c[3]:.0f} m away, {c[4]:.0f} m from goal; reaches him {c[1]*100:.0f}%, chain value {c[2]:.2f} => pass value {c[0]:.2f}")
        bv = cands[0][0]; pc, sc = 0.7 * bv > shot_val, 1.3 * bv < shot_val; verdict = "a pass was clearly better" if pc else ("the shot was clearly better" if sc else "ambiguous (pass value within +-30% of the shot)")
        print(f"   VERDICT: {verdict}. Best pass value {bv:.2f} vs shot value {shot_val:.2f} (ratio {bv/shot_val:.2f}).\n")
    else: print("   No admissible receiver in view. VERDICT: the shot was the only visible option.\n")
