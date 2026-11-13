"""DQ-v5 data: StatsBomb men's competitions >= 2015. Streams gzip in memory; writes reduced tables (shots with freeze frames, completed attacking-half passes with value of the receiver's next shot)."""
import json, os, sys, gzip, time, urllib.request, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import inpoly
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"; xfer = 0; GX = 120.0
def get(path, tries=6):
    global xfer
    for att in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(BASE + path, headers={"Accept-Encoding": "gzip"}), timeout=60); b = r.read(); xfer += len(b)
            return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
        except urllib.error.HTTPError as ex:
            if ex.code == 404: return None
            time.sleep(2 ** att)
        except Exception as ex: print("retry", att, path, ex, flush=True); time.sleep(2 ** att)
    raise RuntimeError(path)
ang = lambda x, y: float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def intri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def secs(e): h, m, s = e["timestamp"].split(":"); return e["period"] * 10000 + int(h) * 3600 + int(m) * 60 + float(s)
comps = [c for c in get("competitions.json") if c["competition_gender"] == "male" and int(c["season_name"][:4]) >= 2015]
TEST = {("55", "2024"), ("223", "2024"), ("9", "2023/2024"), ("44", "2023")}
jobs = []
for c in comps:
    for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json"): jobs.append((f"{c['competition_name']} {c['season_name']}", bool(c.get("match_available_360")), m))
print(len(comps), "competitions,", len(jobs), "matches", flush=True)
shots, passes = [], []; sid = 0
for k, (comp, has360, m) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json")
    if ev is None: continue
    ff = {}
    if has360:
        r = get(f"three-sixty/{mid}.json"); ff = {f["event_uuid"]: f for f in r} if r else {}
    byid = {e["id"]: e for e in ev}; n = len(ev)
    for i, e in enumerate(ev):
        t = e["type"]["name"]
        if e.get("location") is None: continue
        if t == "Shot" and e["shot"]["type"]["name"] == "Open Play":
            s = e["shot"]; fr = s.get("freeze_frame", []); a = np.array(e["location"][:2], float)
            opps = [q["location"][:2] for q in fr if not q["teammate"]]; mates = [q["location"][:2] for q in fr if q["teammate"]]
            gks = [q["location"][:2] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
            if not gks: continue
            kp = byid.get(s.get("key_pass_id")); kpp = kp["pass"] if kp else {}
            O = np.array(opps, float).reshape(-1, 2); dd = np.hypot(*(O - a).T) if len(O) else np.array([30.0])
            gk = np.array(gks[0], float); v = np.array([GX, 40.0]) - a; gl = abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (np.hypot(*v) + 1e-9)
            tech = s["technique"]["name"]
            shots.append(dict(sid=sid, comp=comp, test=any(comp.endswith(x[1]) and str(c_id) == x[0] for x in TEST for c_id in [0]) , match=mid, x=a[0], y=a[1], dist=float(np.hypot(*v)), angle=ang(*a), header=int(s["body_part"]["name"] == "Head"),
                              n_tri=sum(intri(o, a) for o in opps), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), gk_lat=float(gl), n_opp2=int((dd <= 2).sum()), nn_opp=float(dd.min()),
                              first_time=int(bool(s.get("first_time"))), volley=int(tech in ("Volley", "Half Volley", "Lob", "Overhead Kick", "Backheel")),
                              pattern={"From Counter": "counter", "From Corner": "setpiece", "From Free Kick": "setpiece", "From Throw In": "setpiece"}.get(e["play_pattern"]["name"], "other"),
                              prior={"cross": int(bool(kpp.get("cross"))), "through": int(kpp.get("technique", {}).get("name") == "Through Ball"), "cutback": int(bool(kpp.get("cut_back")))} if kp else {"cross": 0, "through": 0, "cutback": 0},
                              sb_xg=s["statsbomb_xg"], goal=int(s["outcome"]["name"] == "Goal"), opps=opps, mates=mates, gk=gks[0], has_prior=int(bool(kp))))
            sid += 1
        if has360 and t == "Pass":
            ps = e["pass"]
            if ("type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" or ps.get("body_part", {}).get("name") == "Head"
                    or "outcome" in ps or ps["end_location"][0] < 60 or e["id"] not in ff or not ps.get("recipient")): continue
            f = ff[e["id"]]; poly = f.get("visible_area"); end = np.array(ps["end_location"][:2], float)
            if not inpoly(end[None], poly)[0]: continue
            fr = f["freeze_frame"]; opps = np.array([q["location"][:2] for q in fr if not q["teammate"]], float).reshape(-1, 2); kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]
            if not kp_ or not len(opps): continue
            gk = np.array(kp_[0], float); dd = np.hypot(*(opps - end).T)
            rid, t0, pos, val, shot_next = ps["recipient"]["id"], secs(e), e["possession"], 0.0, 0
            for j in range(i + 1, min(i + 25, n)):
                q = ev[j]
                if q["possession"] != pos or secs(q) - t0 > 6: break
                if q.get("player", {}).get("id") != rid: continue
                qt = q["type"]["name"]
                if qt == "Shot":
                    if q["shot"]["type"]["name"] == "Open Play": val, shot_next = q["shot"]["statsbomb_xg"], 1
                    break
                if qt in ("Pass", "Dribble", "Dispossessed", "Miscontrol", "Foul Won", "Foul Committed", "Clearance", "Duel", "Block", "Interception"): break
            passes.append(dict(comp=comp, test=False, match=mid, tx=end[0], ty=end[1], dist=float(np.hypot(GX - end[0], 40 - end[1])), angle=ang(*end), n_tri=sum(intri(o, end) for o in opps),
                               gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), n3=int((dd <= 3).sum()), n6=int((dd <= 6).sum()), nn=float(dd.min()), plen=float(ps["length"]), v=val, shot_next=shot_next, hxg=None))
    if k % 100 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, shots {len(shots)}, passes {len(passes)}", flush=True)
Sh, Pa = pd.DataFrame(shots), pd.DataFrame(passes)
tests = ("Euro 2024", "Copa America 2024", "1. Bundesliga 2023/2024", "Major League Soccer 2023", "UEFA Euro 2024")
for df in (Sh, Pa): df["test"] = df.comp.isin(tests) | df.comp.str.contains("Euro 2024|Copa America 2024|Bundesliga 2023/2024|Major League Soccer 2023")
Sh.to_pickle(f"{D}/v5_shots.pkl"); Pa.to_pickle(f"{D}/v5_passes.pkl")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; shots {len(Sh)}, passes {len(Pa)}; comps: {sorted(Sh.comp.unique())}")
