"""Stream StatsBomb open events, reduce to compact tables, delete raw immediately. Budget 1.45 GB."""
import json, os, sys, urllib.request, numpy as np, pandas as pd, random
D = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(D, "raw"); os.makedirs(RAW, exist_ok=True)
BUDGET = 1.45e9; used = 0
pl = json.load(open(os.path.join(D, "m_2_27.json"))); wc = json.load(open(os.path.join(D, "m_43_3.json")))
random.seed(0); wc = random.sample(wc, 40)
jobs = [("PL", m) for m in pl] + [("WC", m) for m in wc]
ON = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*", "Miscontrol", "Dispossessed"}
onball, starts, shots, incomp, meta = [], [], [], [], []
GX, GY = 120.0, 40.0
def ang(x, y):  # goal-mouth angle (radians)
    a = np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x); return float(abs(a))
def intri(px, py, sx, sy):  # point inside triangle shooter-post-post
    def s(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = s(px, py, sx, sy, GX, 36), s(px, py, GX, 36, GX, 44), s(px, py, GX, 44, sx, sy)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
for k, (comp, m) in enumerate(jobs):
    mid = m["match_id"]; f = os.path.join(RAW, f"{mid}.json")
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{mid}.json", f)
    used += os.path.getsize(f); ev = json.load(open(f)); os.remove(f)          # raw deleted immediately
    meta.append(dict(match=mid, comp=comp, date=m["match_date"]))
    # possession outcome: goal scored by possession team within the possession
    goal_pos = {(e["possession"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal" and e["team"]["id"] == e["possession_team"]["id"]}
    pos_first = {}
    for i, e in enumerate(ev):
        p = e["possession"]; t = e["type"]["name"]
        if e.get("location") is None or e["team"]["id"] != e["possession_team"]["id"]: continue
        if t in ON:
            x, y = e["location"][:2]
            onball.append((mid, p, x, y, t, e["play_pattern"]["name"], int(p in goal_pos)))
            if p not in pos_first: pos_first[p] = (x, y, e["play_pattern"]["name"], int(p in goal_pos))
        if t == "Shot":
            s = e["shot"]
            if s["type"]["name"] != "Open Play": continue
            x, y = e["location"][:2]; ff = s.get("freeze_frame", [])
            opp = [q["location"] for q in ff if not q["teammate"]]
            gk = [q["location"] for q in ff if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
            shots.append(dict(match=mid, x=x, y=y, dist=float(np.hypot(GX - x, GY - y)), angle=ang(x, y), header=int(s["body_part"]["name"] == "Head"),
                              n_tri=sum(intri(q[0], q[1], x, y) for q in opp), gk_d=float(np.hypot(GX - gk[0][0], GY - gk[0][1])) if gk else np.nan,
                              sb_xg=s["statsbomb_xg"], goal=int(s["outcome"]["name"] == "Goal")))
        if t == "Pass" and e["pass"].get("outcome", {}).get("name") == "Incomplete" and "end_location" in e["pass"]:
            # next event by the other team with a location: where did the ball actually go?
            nx = next((q for q in ev[i + 1:i + 6] if q["team"]["id"] != e["team"]["id"] and q.get("location")), None)
            el = e["pass"]["end_location"]
            if nx: incomp.append(dict(match=mid, len=e["pass"]["length"], ex=el[0], ey=el[1], ntype=nx["type"]["name"],
                                      d_next=float(np.hypot((120 - nx["location"][0]) - el[0], (80 - nx["location"][1]) - el[1]))))
    for p, (x, y, pp, g) in pos_first.items(): starts.append((mid, p, x, y, pp, g))
    if k % 40 == 0: print(k, comp, f"{used/1e9:.3f} GB", flush=True)
    if used > BUDGET: print("BUDGET STOP at", k); break
compl_len = None
pd.DataFrame(onball, columns=["match", "pos", "x", "y", "type", "pattern", "goal"]).to_pickle(os.path.join(D, "onball.pkl"))
pd.DataFrame(starts, columns=["match", "pos", "x", "y", "pattern", "goal"]).to_pickle(os.path.join(D, "starts.pkl"))
pd.DataFrame(shots).to_pickle(os.path.join(D, "shots.pkl")); pd.DataFrame(incomp).to_pickle(os.path.join(D, "incomp.pkl")); pd.DataFrame(meta).to_pickle(os.path.join(D, "meta.pkl"))
print("DONE matches", len(meta), f"downloaded {used/1e9:.3f} GB; raw dir now holds", len(os.listdir(RAW)), "files")
