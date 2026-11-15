"""DQ-v15 data (PREREG_DQ_v13.txt amendment E): (1) losses of the ball ANYWHERE on the pitch with counter-attack numerical-balance features from the 360 frame; (2) stand-alone Dribble (take-on) events with features from the frame of the PRECEDING event of the same player (no outcome leakage).
Streams every 360 match in memory; writes the reduced tables v15_loss.pkl and v15_dribble.pkl (deleted after the fit)."""
import json, os, sys, gzip, time, urllib.request, numpy as np, pandas as pd
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
    """Context of point p (attacking toward x = 120) incl. the counter numerical balance: opponents beyond the ball (x < p.x) against our players between the ball and our goal."""
    do = np.hypot(*(O - p).T) if len(O) else np.array([99.0]); dt = np.hypot(*(T - p).T) if len(T) else np.array([99.0])
    ahead = O[O[:, 0] < p[0]] if len(O) else np.zeros((0, 2)); cover = int((T[:, 0] < p[0]).sum()) if len(T) else 0
    free = [o for o in ahead if not len(T) or np.hypot(*(T - o).T).min() > 6.0]
    return dict(n_opp2=int((do <= 2).sum()), n_opp3=int((do <= 3).sum()), n_opp6=int((do <= 6).sum()), nn_opp=float(min(do.min(), 30)), n_mate10=int((dt <= 10).sum()), nn_mate=float(min(dt.min(), 30)),
                n_opp_ahead=len(ahead), n_cover=cover, n_free_run=len(free), run_x=float(ahead[:, 0].min()) if len(ahead) else 120.0, nn_run=float(min(np.hypot(*(np.array(free) - p).T).min(), 30)) if free else 30.0)
FAILS = {"Incomplete", "Out", "Pass Offside", "Unknown"}
comps = [c for c in get("competitions.json") if c["competition_gender"] == "male" and int(c["season_name"][:4]) >= 2015 and c.get("match_available_360")]
jobs = [(f"{c['competition_name']} {c['season_name']}", m) for c in comps for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json")]
print(len(comps), "competitions,", len(jobs), "matches", flush=True)
loss, drb = [], []
for k, (comp, m) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json"); r = get(f"three-sixty/{mid}.json")
    if ev is None or not r: continue
    ff = {f["event_uuid"]: f for f in r}; n = len(ev); shots = [(secs(e), e["team"]["id"], e["shot"]["statsbomb_xg"], e["possession"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play"]
    def frame(e):
        fr = ff[e["id"]]["freeze_frame"]; O = np.array([q["location"][:2] for q in fr if not q["teammate"]], float).reshape(-1, 2); T = np.array([q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")], float).reshape(-1, 2)
        kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]; return O, T, (np.array(kp_[0], float) if kp_ else None)
    for i, e in enumerate(ev):
        t = e["type"]["name"]; loc = e.get("location")
        if loc is None or e["id"] not in ff: continue
        O, T, gk = frame(e); a = np.array(loc[:2], float); t0 = secs(e); tm = e["team"]["id"]
        if not len(O): continue
        own10 = sum(x for tt, ti, x, pp in shots if ti == tm and t0 < tt <= t0 + H); opp10 = sum(x for tt, ti, x, pp in shots if ti != tm and t0 < tt <= t0 + H)
        base = dict(comp=comp, match=mid, time=t0, period=e["period"], minute=e["minute"], second=e["second"], team=e["team"]["name"], index=e["index"], ax=a[0], ay=a[1], own10=own10, opp10=opp10)
        lp, lkind = None, None
        if t == "Pass" and e["pass"].get("outcome", {}).get("name") in FAILS and "end_location" in e["pass"]: lp, lkind = np.array(e["pass"]["end_location"][:2], float), "pass"
        elif t == "Dribble" and e["dribble"].get("outcome", {}).get("name") == "Incomplete": lp, lkind = a, "dribble"
        elif t in ("Dispossessed", "Miscontrol"): lp, lkind = a, t.lower()
        if lp is not None and lp[0] >= 0:
            loss.append(dict(base, kind=lkind, lx=lp[0], ly=lp[1], dist_own=float(np.hypot(lp[0], lp[1] - 40)), ang_own=float(abs(np.arctan2(44 - lp[1], lp[0]) - np.arctan2(36 - lp[1], lp[0]))), **around(lp, O, T)))
        if t == "Dribble" and a[0] >= 70:
            # frame of the PRECEDING event of the same player (Ball Receipt* or Carry, same possession, at most 4 s before): the frame is taken BEFORE the duel
            prev = None
            for j in range(i - 1, max(-1, i - 10), -1):
                q = ev[j]
                if q.get("player", {}).get("id") != e["player"]["id"] or q["possession"] != e["possession"]: continue
                if q["type"]["name"] in ("Carry", "Ball Receipt*") and q["id"] in ff and q.get("location") and 0 <= t0 - secs(q) <= 4.0: prev = q; break
            if prev is None: continue
            Op, Tp, gkp = frame(prev); p0 = np.array(prev["location"][:2], float)
            if not len(Op) or gkp is None: continue
            dv = np.array([GX - p0[0], 40 - p0[1]]); dn = np.hypot(*dv); u = dv / max(dn, 1e-9); rel = Op - p0; dd = np.hypot(*rel.T); cosang = (rel @ u) / np.maximum(dd, 1e-9); cone = (cosang >= np.cos(np.pi / 4)) & (dd <= 12.0) & (dd > 0.1)
            ok = int(e["dribble"].get("outcome", {}).get("name") == "Complete"); ctx = around(p0, Op, Tp)
            nn_cone = float(dd[cone].min()) if cone.any() else 15.0
            opp_c = Op[cone][np.argmin(dd[cone])] if cone.any() else None
            drb.append(dict(base, ok=ok, px=p0[0], py=p0[1], dt_prev=float(t0 - secs(prev)), prev_kind=prev["type"]["name"], dist=float(np.hypot(GX - p0[0], 40 - p0[1])), angle=ang(*p0), n_tri=sum(intri(o, p0) for o in Op), gk_d=float(np.hypot(GX - gkp[0], 40 - gkp[1])),
                            n3=ctx["n_opp3"], n6=ctx["n_opp6"], nn=ctx["nn_opp"], n_cone=int(cone.sum()), nn_cone=nn_cone, n_mate10=ctx["n_mate10"], dest_x=float(opp_c[0] + 2 * u[0]) if opp_c is not None else np.nan, dest_y=float(opp_c[1] + 2 * u[1]) if opp_c is not None else np.nan))
    if k % 100 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, losses {len(loss)}, take-ons {len(drb)}", flush=True)
Lo, Dr = pd.DataFrame(loss), pd.DataFrame(drb)
for df in (Lo, Dr): df["test"] = df.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024")
Lo.to_pickle(f"{D}/v15_loss.pkl"); Dr.to_pickle(f"{D}/v15_dribble.pkl")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; losses {len(Lo)}; take-ons {len(Dr)} (success {Dr.ok.mean()*100:.1f}%)")
