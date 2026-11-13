"""DQ-v1 data: gzip transfer, decompressed in memory only (nothing raw written to disk). Reduced tables only."""
import json, os, gzip, urllib.request, random, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
xfer = 0; BUDGET = 1.0e9
import time
def get(path):
    global xfer
    for att in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(BASE + path, headers={"Accept-Encoding": "gzip"}), timeout=60); b = r.read(); break
        except Exception as ex:
            print("retry", att, path, ex, flush=True); time.sleep(2 ** att)
    else: raise RuntimeError("download failed " + path)
    xfer += len(b)
    return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
pl, wc18, eu, wc22 = get("matches/2/27.json"), get("matches/43/3.json"), get("matches/55/43.json"), get("matches/43/106.json")
random.seed(0); wc18 = random.sample(wc18, 40)                 # identical draw to v0
jobs = [("PL", m, False) for m in pl] + [("WC18", m, False) for m in wc18] + [("EU20", m, True) for m in eu] + [("WC22", m, True) for m in wc22]
ON = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*", "Miscontrol", "Dispossessed"}; GX = 120.0
def ang(x, y): return float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def intri(px, py, sx, sy):
    def s(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = s(px, py, sx, sy, GX, 36), s(px, py, GX, 36, GX, 44), s(px, py, GX, 44, sx, sy)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def segd(p, a, b):
    ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab + 1e-12), 0, 1); return np.hypot(*(a + t[:, None] * ab - p).T) if len(p) else np.array([])
onball, starts, shots, passes, meta = [], [], [], [], []; actor_err = []
for k, (comp, m, has360) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json"); ff = {f["event_uuid"]: f for f in get(f"three-sixty/{mid}.json")} if has360 else {}
    meta.append(dict(match=mid, comp=comp, date=m["match_date"]))
    goal_pos = {e["possession"] for e in ev if e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal" and e["team"]["id"] == e["possession_team"]["id"]}
    pos_first = {}
    for e in ev:
        p, t = e["possession"], e["type"]["name"]
        if e.get("location") is None or e["team"]["id"] != e["possession_team"]["id"]: continue
        x, y = e["location"][:2]
        if t in ON:
            onball.append((mid, p, x, y, int(p in goal_pos)))
            if p not in pos_first: pos_first[p] = (x, y, e["play_pattern"]["name"], int(p in goal_pos))
        if t == "Shot" and e["shot"]["type"]["name"] == "Open Play":
            s = e["shot"]; fr = s.get("freeze_frame", []); opp = [q["location"] for q in fr if not q["teammate"]]
            gk = [q["location"] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
            shots.append(dict(match=mid, dist=float(np.hypot(GX - x, 40 - y)), angle=ang(x, y), header=int(s["body_part"]["name"] == "Head"),
                              n_tri=sum(intri(q[0], q[1], x, y) for q in opp), gk_d=float(np.hypot(GX - gk[0][0], 40 - gk[0][1])) if gk else np.nan, sb_xg=s["statsbomb_xg"], goal=int(s["outcome"]["name"] == "Goal")))
        if has360 and t == "Pass":
            ps = e["pass"]; oc = ps.get("outcome", {}).get("name")
            if "type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" \
               or ps.get("body_part", {}).get("name") == "Head" or ps["length"] < 5 or oc not in (None, "Incomplete"): continue
            row = dict(match=mid, comp=comp, complete=int(oc is None), length=ps["length"], has360=e["id"] in ff)
            if row["has360"]:
                fr = ff[e["id"]]["freeze_frame"]; a = np.array([x, y]); end = np.array(ps["end_location"][:2])
                act = [q["location"] for q in fr if q.get("actor")]
                if act: actor_err.append(float(np.hypot(*(np.array(act[0]) - a))))
                mates = np.array([q["location"] for q in fr if q["teammate"] and not q.get("actor")]).reshape(-1, 2); opps = np.array([q["location"] for q in fr if not q["teammate"]]).reshape(-1, 2)
                row.update(n_mates=len(mates), n_opps=len(opps))
                dvec = end - a; th = np.arctan2(dvec[1], dvec[0]); ok = np.hypot(*(mates - a).T) >= 3 if len(mates) else np.array([], bool)
                cand = mates[ok]; dev = np.degrees(np.abs((np.arctan2(*(cand - a).T[::-1]) - th + np.pi) % (2 * np.pi) - np.pi)) if len(cand) else np.array([])
                o = np.argsort(dev); ident = len(cand) > 0 and dev[o[0]] <= 15 and (len(cand) == 1 or dev[o[1]] - dev[o[0]] >= 5)
                row["ident"] = int(ident); row["no_cone"] = int(len(cand) == 0 or dev[o[0]] > 15)
                if len(mates): row["near_end_is_ident"] = int(ident and np.allclose(cand[o[0]], mates[np.argmin(np.hypot(*(mates - end).T))]))
                if ident:
                    tg = cand[o[0]]; v = tg - a
                    row.update(t_dist=float(np.hypot(*v)), t_ang=float(abs(np.arctan2(v[1], v[0]))), t_fwd=float(v[0]),
                               lane_min=float(segd(opps, a, tg).min()) if len(opps) else 30.0, opp5=int((np.hypot(*(opps - tg).T) <= 5).sum()) if len(opps) else 0)
            passes.append(row)
    for p, (x, y, pp, g) in pos_first.items(): starts.append((mid, p, x, y, pp, g))
    if k % 50 == 0: print(k, comp, f"transferred {xfer/1e6:.0f} MB", flush=True)
    if xfer > BUDGET: print("BUDGET STOP", k); break
pd.DataFrame(onball, columns=["match", "pos", "x", "y", "goal"]).to_pickle(f"{D}/onball.pkl"); pd.DataFrame(starts, columns=["match", "pos", "x", "y", "pattern", "goal"]).to_pickle(f"{D}/starts.pkl")
pd.DataFrame(shots).to_pickle(f"{D}/shots.pkl"); pd.DataFrame(passes).to_pickle(f"{D}/passes360.pkl"); pd.DataFrame(meta).to_pickle(f"{D}/meta.pkl")
print(f"actor vs event location: median {np.median(actor_err):.2f} units, 95th pct {np.percentile(actor_err, 95):.2f} (orientation check)")
print("DONE matches", len(meta), f"transferred {xfer/1e6:.0f} MB (gzip); nothing raw written to disk")
