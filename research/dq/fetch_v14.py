"""DQ-v14 validation data: every foot shot, ground pass and carry at x >= 80 in the 360 competitions + Copa America 2024 (shots), with the full situation and the realised xG of the next 10 s for and against."""
import json, os, sys, gzip, time, urllib.request, collections, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"; xfer = 0; GX = 120.0; H = 10.0
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
def around(p, O, T):
    """Context of point p: opponents / teammates near it and relative to it (attacking toward x = 120)."""
    do = np.hypot(*(O - p).T) if len(O) else np.array([99.0]); dt = np.hypot(*(T - p).T) if len(T) else np.array([99.0])
    return dict(n_opp2=int((do <= 2).sum()), n_opp3=int((do <= 3).sum()), n_opp6=int((do <= 6).sum()), nn_opp=float(do.min()), n_mate10=int((dt <= 10).sum()), nn_mate=float(dt.min()),
                n_opp_ahead=int((O[:, 0] < p[0]).sum()) if len(O) else 0, n_cover=int((T[:, 0] < p[0]).sum()) if len(T) else 0)
FAILS = {"Incomplete", "Out", "Pass Offside", "Unknown"}
comps = [c for c in get("competitions.json") if c["competition_gender"] == "male" and int(c["season_name"][:4]) >= 2015 and (c.get("match_available_360") or (c["competition_id"] == 223 and c["season_name"] == "2024"))]
jobs = [(f"{c['competition_name']} {c['season_name']}", bool(c.get('match_available_360')), m) for c in comps for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json")]
print(len(comps), "competitions,", len(jobs), "matches", flush=True)
dec = []
for k, (comp, has360, m) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json")
    if ev is None: continue
    r = get(f"three-sixty/{mid}.json") if has360 else None; ff = {f["event_uuid"]: f for f in r} if r else {}
    shots = [(secs(e), e["team"]["id"], e["shot"]["statsbomb_xg"], e["possession"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play"]; n = len(ev)
    for i, e in enumerate(ev):
        t = e["type"]["name"]; loc = e.get("location")
        if loc is None or loc[0] < 80: continue
        a = np.array(loc[:2], float); t0 = secs(e); tm = e["team"]["id"]; row = None
        if t == "Shot" and e["shot"]["type"]["name"] == "Open Play" and e["shot"]["body_part"]["name"] != "Head":
            fr = e["shot"].get("freeze_frame", []); O = [q["location"][:2] for q in fr if not q["teammate"]]; Tm = [q["location"][:2] for q in fr if q["teammate"]]; gks = [q["location"][:2] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
            if gks and O: row = dict(kind="shot", tx=np.nan, ty=np.nan, sb_xg=e["shot"]["statsbomb_xg"], goal=int(e["shot"]["outcome"]["name"] == "Goal"), opps=O, mates=Tm, gk=gks[0])
        elif has360 and e["id"] in ff and t in ("Pass", "Carry"):
            fr = ff[e["id"]]["freeze_frame"]; O = [q["location"][:2] for q in fr if not q["teammate"]]; Tm = [q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")]; kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]
            if not kp_ or not O: continue
            if t == "Pass":
                ps = e["pass"]
                if "type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" or ps.get("body_part", {}).get("name") == "Head" or "end_location" not in ps: continue
                row = dict(kind="pass", tx=ps["end_location"][0], ty=ps["end_location"][1], completed=int("outcome" not in ps), opps=O, mates=Tm, gk=kp_[0])
            elif "end_location" in e["carry"] and float(np.hypot(*(np.array(e["carry"]["end_location"][:2]) - a))) >= 1.0:
                nxt = None
                for j in range(i + 1, min(i + 6, n)):
                    q = ev[j]
                    if q.get("player", {}).get("id") != e["player"]["id"] or q["type"]["name"] in ("Ball Receipt*", "Pressure", "Carry"): continue
                    nxt = q; break
                fail = int(nxt is not None and secs(nxt) - t0 < 6 and (nxt["type"]["name"] in ("Dispossessed", "Miscontrol") or (nxt["type"]["name"] == "Dribble" and nxt["dribble"].get("outcome", {}).get("name") == "Incomplete")))
                row = dict(kind="carry", tx=e["carry"]["end_location"][0], ty=e["carry"]["end_location"][1], fail=fail, opps=O, mates=Tm, gk=kp_[0])
        if row is None: continue
        own10 = sum(x for tt, ti, x, pp in shots if ti == tm and t0 < tt <= t0 + H); opp10 = sum(x for tt, ti, x, pp in shots if ti != tm and t0 < tt <= t0 + H); chain10 = sum(x for tt, ti, x, pp in shots if ti == tm and pp == e["possession"] and t0 < tt <= t0 + H)
        dec.append(dict(row, comp=comp, match=mid, period=e["period"], minute=e["minute"], second=e["second"], team=e["team"]["name"], index=e["index"], ax=a[0], ay=a[1], own10=own10, opp10=opp10, chain10=chain10))
    if k % 100 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, decisions {len(dec)}", flush=True)
Dd = pd.DataFrame(dec); Dd["test"] = Dd.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024"); Dd.to_pickle(f"{D}/v14_dec.pkl")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; decisions {len(Dd)}: {dict(Dd.kind.value_counts())}; test {int(Dd.test.sum())}")
