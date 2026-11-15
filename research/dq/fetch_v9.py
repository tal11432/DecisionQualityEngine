"""DQ-v9 data (events for all men's comps >= 2015; 360 for 360 comps). Tables: base (all-matches pass baseline), p360 (360 pass population with new features), shots (all, light) and tshots (test shots with frames)."""
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
inbox = lambda P: ((P[:, 0] >= 102) & (P[:, 1] >= 18) & (P[:, 1] <= 62)) if len(P) else np.array([], bool)
pat = lambda e: "counter" if e["play_pattern"]["name"] == "From Counter" else ("set" if e["play_pattern"]["name"] in ("From Corner", "From Free Kick", "From Throw In", "From Kick Off", "From Goal Kick") else "other")
comps = [c for c in get("competitions.json") if c["competition_gender"] == "male" and int(c["season_name"][:4]) >= 2015]
jobs = [(f"{c['competition_name']} {c['season_name']}", bool(c.get("match_available_360")), m) for c in comps for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json")]
print(len(comps), "competitions,", len(jobs), "matches", flush=True)
base, p360, shots = [], [], []
TESTRE = "Euro 2024|Copa America 2024|Bundesliga 2023/2024|Major League Soccer 2023"
for k, (comp, has360, m) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json")
    if ev is None: continue
    ff = {}
    if has360:
        r = get(f"three-sixty/{mid}.json"); ff = {f["event_uuid"]: f for f in r} if r else {}
    n = len(ev); sp = collections.defaultdict(list); pstart = {}; cnt = collections.Counter(); nprev = [0] * n
    for i, e in enumerate(ev):
        pstart.setdefault(e["possession"], secs(e))
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play": sp[(e["possession"], e["team"]["id"])].append((secs(e), e["shot"]["statsbomb_xg"]))
        nprev[i] = cnt[(e["possession"], e["team"]["id"])]
        if e["type"]["name"] == "Pass": cnt[(e["possession"], e["team"]["id"])] += 1
    for i, e in enumerate(ev):
        t = e["type"]["name"]
        if e.get("location") is None: continue
        t0 = secs(e); tsince = t0 - pstart[e["possession"]]
        if t == "Shot" and e["shot"]["type"]["name"] == "Open Play":
            s = e["shot"]; fr = s.get("freeze_frame", []); a = np.array(e["location"][:2], float)
            opps = [q["location"][:2] for q in fr if not q["teammate"]]; mates = [q["location"][:2] for q in fr if q["teammate"]]; gks = [q["location"][:2] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
            if not gks: continue
            O = np.array(opps, float).reshape(-1, 2); dd = np.hypot(*(O - a).T) if len(O) else np.array([30.0]); v = np.array([GX, 40.0]) - a; gk = np.array(gks[0], float); dist = float(np.hypot(*v)); an = ang(*a)
            cont = sum(x for tt, x in sp[(e["possession"], e["team"]["id"])] if t0 < tt <= t0 + H); istest = any(x in comp for x in TESTRE.split("|"))
            row = dict(comp=comp, match=mid, x=a[0], y=a[1], dist=dist, angle=an, header=int(s["body_part"]["name"] == "Head"), n_tri=sum(intri(o, a) for o in opps), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])),
                       gk_lat=float(abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9)), n_opp2=int((dd <= 2).sum()), nn_opp=float(dd.min()), da=dist * an, ldist=float(np.log(dist)), cont=cont, sb_xg=s["statsbomb_xg"], goal=int(s["outcome"]["name"] == "Goal"),
                       pattern=pat(e), nprev=nprev[i], tsince=tsince)
            if istest and True: row.update(opps=opps, mates=mates, gk=gks[0])
            shots.append(row)
        if t == "Pass":
            ps = e["pass"]
            if ("type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" or ps.get("body_part", {}).get("name") == "Head" or "outcome" in ps
                    or ps["end_location"][0] < 60 or not ps.get("recipient")): continue
            a = np.array(e["location"][:2], float); end = np.array(ps["end_location"][:2], float); chain = sum(x for tt, x in sp[(e["possession"], e["team"]["id"])] if t0 < tt <= t0 + H)
            nsh = sum(1 for tt, x in sp[(e["possession"], e["team"]["id"])] if t0 < tt <= t0 + H); pt = pat(e)
            base.append(dict(comp=comp, match=mid, sx=a[0], sy=a[1], tx=end[0], ty=end[1], plen=float(ps["length"]), counter=int(pt == "counter"), setp=int(pt == "set"), nprev=nprev[i], tsince=tsince, chain=chain, nsh=nsh))
            if has360 and e["id"] in ff:
                f = ff[e["id"]]
                if not inpoly(end[None], f.get("visible_area"))[0]: continue
                fr = f["freeze_frame"]; O = np.array([q["location"][:2] for q in fr if not q["teammate"]], float).reshape(-1, 2); T_ = np.array([q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")], float).reshape(-1, 2)
                kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]
                if not kp_ or not len(O): continue
                gk = np.array(kp_[0], float); dd = np.hypot(*(O - end).T); dist = float(np.hypot(GX - end[0], 40 - end[1])); an = ang(*end)
                rec = dict(nn_rec=np.nan, n3_rec=np.nan, n6_rec=np.nan)
                for j in range(i + 1, min(i + 6, n)):
                    q = ev[j]
                    if q["type"]["name"] == "Ball Receipt*" and q.get("player", {}).get("id") == ps["recipient"]["id"] and q["id"] in ff:
                        fr2 = ff[q["id"]]["freeze_frame"]; O2 = np.array([z["location"][:2] for z in fr2 if not z["teammate"]], float).reshape(-1, 2)
                        if len(O2): d2 = np.hypot(*(O2 - np.array(q["location"][:2])).T); rec = dict(nn_rec=float(d2.min()), n3_rec=int((d2 <= 3).sum()), n6_rec=int((d2 <= 6).sum()))
                        break
                tmd = np.hypot(*(T_ - end).T) if len(T_) else np.array([99.0])
                p360.append(dict(comp=comp, match=mid, tx=end[0], dist=dist, angle=an, n_tri=sum(intri(o, end) for o in O), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), n3=int((dd <= 3).sum()), n6=int((dd <= 6).sum()), nn=float(dd.min()),
                                 plen=float(ps["length"]), fwd=float(end[0] - a[0]), pdist=float(np.hypot(GX - a[0], 40 - a[1])), counter=int(pt == "counter"), setp=int(pt == "set"), nprev=nprev[i], tsince=tsince,
                                 att_box=int(inbox(T_).sum()), def_box=int(inbox(O).sum()), sup10=int((tmd <= 10).sum()), ahead25=int(((T_[:, 0] > end[0]) & (tmd <= 25)).sum()) if len(T_) else 0,
                                 opp_between=int(((O[:, 0] > end[0]) & (np.abs(O[:, 1] - end[1]) <= 12)).sum()), opp5_passer=int((np.hypot(*(O - a).T) <= 5).sum()), chain=chain, nsh=nsh, **rec))
    if k % 100 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, base {len(base)}, p360 {len(p360)}, shots {len(shots)}", flush=True)
B, P, S = pd.DataFrame(base), pd.DataFrame(p360), pd.DataFrame(shots)
for df in (B, P, S): df["test"] = df.comp.str.contains(TESTRE)
B.to_pickle(f"{D}/v9_base.pkl"); P.to_pickle(f"{D}/v9_p360.pkl"); S.to_pickle(f"{D}/v9_shots.pkl")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; base {len(B)}, p360 {len(P)}, shots {len(S)}")
