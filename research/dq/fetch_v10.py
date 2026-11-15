"""DQ-v10 data: shots (all men's comps >= 2015) with continuation and context; decision-moment passes in 360 comps with chain value."""
import json, os, sys, gzip, time, urllib.request, collections, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import inpoly
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"; xfer = 0; GX = 120.0; H = 10.0
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
def ctx(a, opps, gk):
    O = np.array(opps, float).reshape(-1, 2); dd = np.hypot(*(O - a).T) if len(O) else np.array([30.0]); v = np.array([GX, 40.0]) - a; dist = float(np.hypot(*v)); an = ang(*a)
    return dict(x=a[0], y=a[1], dist=dist, angle=an, n_tri=sum(intri(o, a) for o in opps), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), gk_lat=float(abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9)),
                n_opp2=int((dd <= 2).sum()), nn_opp=float(dd.min()), da=dist * an, ldist=float(np.log(dist)))
comps = [c for c in get("competitions.json") if c["competition_gender"] == "male" and int(c["season_name"][:4]) >= 2015 and c.get("match_available_360")]
jobs = [(f"{c['competition_name']} {c['season_name']}", bool(c.get("match_available_360")), m) for c in comps for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json")]
print(len(comps), "competitions,", len(jobs), "matches", flush=True)
shots, moments = [], []
for k, (comp, has360, m) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json")
    if ev is None: continue
    ff = {}
    if has360:
        r = get(f"three-sixty/{mid}.json"); ff = {f["event_uuid"]: f for f in r} if r else {}
    byid = {e["id"]: e for e in ev}; n = len(ev); sp = collections.defaultdict(list)
    for e in ev:
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play": sp[(e["possession"], e["team"]["id"])].append((secs(e), e["shot"]["statsbomb_xg"]))
    for i, e in enumerate(ev):
        t = e["type"]["name"]
        if e.get("location") is None: continue
        if t == "Shot" and e["shot"]["type"]["name"] == "Open Play":
            s = e["shot"]; fr = s.get("freeze_frame", []); a = np.array(e["location"][:2], float); t0 = secs(e)
            opps = [q["location"][:2] for q in fr if not q["teammate"]]; gks = [q["location"][:2] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
            if not gks: continue
            kp = byid.get(s.get("key_pass_id")); kpp = kp["pass"] if kp else {}; tech = s["technique"]["name"]; cont = sum(x for tt, x in sp[(e["possession"], e["team"]["id"])] if t0 < tt <= t0 + H)
            row = ctx(a, opps, np.array(gks[0], float)); row.update(comp=comp, match=mid, header=int(s["body_part"]["name"] == "Head"), first_time=int(bool(s.get("first_time"))), volley=int(tech in ("Volley", "Half Volley", "Lob", "Overhead Kick", "Backheel")),
                pat_counter=int(e["play_pattern"]["name"] == "From Counter"), pat_set=int(e["play_pattern"]["name"] in ("From Corner", "From Free Kick", "From Throw In")),
                prior_cross=int(bool(kpp.get("cross"))), prior_through=int(kpp.get("technique", {}).get("name") == "Through Ball"), prior_cutback=int(bool(kpp.get("cut_back"))), sb_xg=s["statsbomb_xg"], goal=int(s["outcome"]["name"] == "Goal"), cont=cont, has360=has360)
            pass
        if has360 and t == "Pass" and e["location"][0] >= 90 and e["id"] in ff:
            ps = e["pass"]
            if ("type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" or ps.get("body_part", {}).get("name") == "Head" or "end_location" not in ps): continue
            fr = ff[e["id"]]["freeze_frame"]; opps = [q["location"][:2] for q in fr if not q["teammate"]]; kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]
            mates = [q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")]
            if not kp_ or not opps: continue
            a = np.array(e["location"][:2], float); t0 = secs(e); chain = sum(x for tt, x in sp[(e["possession"], e["team"]["id"])] if t0 < tt <= t0 + H)
            row = ctx(a, opps, np.array(kp_[0], float)); row.update(comp=comp, match=mid, v_chain=chain, completed=int("outcome" not in ps), end=ps["end_location"][:2], opps=opps, mates=mates, gk=kp_[0], time=t0)
            moments.append(row)
    if k % 100 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, shots {len(shots)}, pass moments {len(moments)}", flush=True)
Sh, Mo = pd.DataFrame(shots), pd.DataFrame(moments)
for df in (Sh, Mo): df["test"] = df.comp.str.contains("Euro 2024|Copa America 2024|Bundesliga 2023/2024|Major League Soccer 2023")
Sh.to_pickle(f"{D}/v10_shots.pkl"); Mo.to_pickle(f"{D}/v10_moments.pkl")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; shots {len(Sh)}, pass moments {len(Mo)}")
