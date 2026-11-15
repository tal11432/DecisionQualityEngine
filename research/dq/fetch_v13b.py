"""DQ-v13b data: losses of the ball ANYWHERE on the pitch (x >= 0) and one-on-one shots in the 360 competitions, with the 360 frame context and the realised xG of the next 10 s for and against.
Streams gzip in memory; writes reduced tables (v13b_acts_unused.pkl, v13_loss.pkl)."""
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
comps = [c for c in get("competitions.json") if c["competition_gender"] == "male" and int(c["season_name"][:4]) >= 2015 and c.get("match_available_360")]
jobs = [(f"{c['competition_name']} {c['season_name']}", m) for c in comps for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json")]
print(len(comps), "competitions,", len(jobs), "matches", flush=True)
acts, loss, oneone = [], [], []
for k, (comp, m) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json"); r = get(f"three-sixty/{mid}.json")
    if ev is None or not r: continue
    oneone += [(comp, e["shot"]["statsbomb_xg"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play" and e["shot"].get("one_on_one")]
    ff = {f["event_uuid"]: f for f in r}; n = len(ev); shots = [(secs(e), e["team"]["id"], e["shot"]["statsbomb_xg"], e["possession"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play"]
    for i, e in enumerate(ev):
        t = e["type"]["name"]; loc = e.get("location")
        if loc is None or e["id"] not in ff: continue
        fr = ff[e["id"]]["freeze_frame"]; O = np.array([q["location"][:2] for q in fr if not q["teammate"]], float).reshape(-1, 2); T = np.array([q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")], float).reshape(-1, 2)
        kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]; a = np.array(loc[:2], float); t0 = secs(e); tm = e["team"]["id"]
        if not kp_ or not len(O): continue
        gk = np.array(kp_[0], float); own10 = sum(x for tt, ti, x, pp in shots if ti == tm and t0 < tt <= t0 + H); opp10 = sum(x for tt, ti, x, pp in shots if ti != tm and t0 < tt <= t0 + H)
        chain10 = sum(x for tt, ti, x, pp in shots if ti == tm and pp == e["possession"] and t0 < tt <= t0 + H)
        base = dict(comp=comp, match=mid, time=t0, period=e["period"], minute=e["minute"], second=e["second"], team=e["team"]["name"], index=e["index"], ax=a[0], ay=a[1], own10=own10, opp10=opp10, chain10=chain10)
        # ---- losses of the ball
        lp, lkind = None, None
        if t == "Pass" and e["pass"].get("outcome", {}).get("name") in FAILS and "end_location" in e["pass"]: lp, lkind = np.array(e["pass"]["end_location"][:2], float), "pass"
        elif t == "Dribble" and e["dribble"].get("outcome", {}).get("name") == "Incomplete": lp, lkind = a, "dribble"
        elif t in ("Dispossessed", "Miscontrol"): lp, lkind = a, t.lower()
        if lp is not None and lp[0] >= 0:
            ctxl = around(lp, O, T); xa_ = a
            loss.append(dict(base, kind=lkind, lx=lp[0], ly=lp[1], dist_own=float(np.hypot(lp[0], lp[1] - 40)), ang_own=float(abs(np.arctan2(44 - lp[1], lp[0]) - np.arctan2(36 - lp[1], lp[0]))), **ctxl))
        # ---- carries and dribbles in the attacking third
        if False and t == "Carry" and a[0] >= 75 and "end_location" in e["carry"]:
            end = np.array(e["carry"]["end_location"][:2], float); nxt = None
            for j in range(i + 1, min(i + 6, n)):
                q = ev[j]
                if q.get("player", {}).get("id") != e["player"]["id"] or q["type"]["name"] in ("Ball Receipt*", "Pressure", "Carry"): continue
                nxt = q; break
            fail = int(nxt is not None and secs(nxt) - t0 < 6 and (nxt["type"]["name"] in ("Dispossessed", "Miscontrol") or (nxt["type"]["name"] == "Dribble" and nxt["dribble"].get("outcome", {}).get("name") == "Incomplete")))
            L = float(np.hypot(*(end - a)))
            if L >= 1.0:
                dest = around(end, O, T); st = around(a, O, T); dist_a = float(np.hypot(GX - a[0], 40 - a[1]))
                acts.append(dict(base, kind="carry", ex=end[0], ey=end[1], L=L, fail=fail, dist=float(np.hypot(GX - end[0], 40 - end[1])), angle=ang(*end), n_tri=sum(intri(o, end) for o in O), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])),
                                 n3=dest["n_opp3"], n6=dest["n_opp6"], nn=dest["nn_opp"], fwd=float(end[0] - a[0]), pdist=dist_a, s_n_opp3=st["n_opp3"], s_nn_opp=st["nn_opp"], s_n_opp6=st["n_opp6"]))
        if False and t == "Dribble" and a[0] >= 75:
            ok = int(e["dribble"].get("outcome", {}).get("name") == "Complete"); end = a
            if ok:
                for j in range(i + 1, min(i + 6, n)):
                    q = ev[j]
                    if q.get("player", {}).get("id") == e["player"]["id"] and q["type"]["name"] not in ("Ball Receipt*", "Pressure", "Carry", "Dribble") and q.get("location"): end = np.array(q["location"][:2], float); break
            st = around(a, O, T); dist_a = float(np.hypot(GX - a[0], 40 - a[1]))
            acts.append(dict(base, kind="dribble", complete=ok, ex=end[0], ey=end[1], L=float(np.hypot(*(end - a))), dist=float(np.hypot(GX - end[0], 40 - end[1])), angle=ang(*end), n_tri=sum(intri(o, end) for o in O), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])),
                             n3=int((np.hypot(*(O - end).T) <= 3).sum()), n6=int((np.hypot(*(O - end).T) <= 6).sum()), nn=float(np.hypot(*(O - end).T).min()), fwd=float(end[0] - a[0]), pdist=dist_a, s_n_opp3=st["n_opp3"], s_nn_opp=st["nn_opp"], s_n_opp6=st["n_opp6"]))
    if k % 100 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, actions {len(acts)}, losses {len(loss)}", flush=True)
A, Lo = pd.DataFrame(acts, columns=["comp"]), pd.DataFrame(loss); O1 = pd.DataFrame(oneone, columns=["comp", "xg"]); O1.to_pickle(f"{D}/v13_oneone.pkl")
for df in (A, Lo): df["test"] = df.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024")
A.to_pickle(f"{D}/v13b_acts_unused.pkl"); Lo.to_pickle(f"{D}/v13_loss.pkl")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; losses {len(Lo)}; one-on-one shots {len(O1)}")
