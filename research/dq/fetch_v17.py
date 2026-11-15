"""DQ-v17 data (PREREG_DQ_v13.txt amendment M, the 20 s horizon experiment): the same decisions as fetch_v16.py with the realised xG of the next 10 s AND 20 s (own, opponent, own chain, and for shots the continuation), plus every loss of the ball anywhere with its context and the 10 s / 20 s outcomes.
Derived from DQ-v16 data (PREREG_DQ_v13.txt amendment J): every open-play foot shot at x >= 60, ground pass and carry (>= 1 m) ANYWHERE on the pitch in the 360 competitions (+ Copa America 2024 for shots), with the full 360 situation
(the opposing keeper is NOT required) and the realised xG of the next 10 s for and against. Streams gzip in memory; writes the reduced tables v17_dec.pkl and v17_loss.pkl (deleted after the fit)."""
import json, os, sys, gzip, time, urllib.request, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"; xfer = 0; H = 10.0; H2 = 20.0
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
def secs(e): h, m, s = e["timestamp"].split(":"); return e["period"] * 10000 + int(h) * 3600 + int(m) * 60 + float(s)
comps = [c for c in get("competitions.json") if c["competition_gender"] == "male" and int(c["season_name"][:4]) >= 2015 and (c.get("match_available_360") or (c["competition_id"] == 223 and c["season_name"] == "2024"))]
jobs = [(f"{c['competition_name']} {c['season_name']}", bool(c.get('match_available_360')), m) for c in comps for m in get(f"matches/{c['competition_id']}/{c['season_id']}.json")]
print(len(comps), "competitions,", len(jobs), "matches", flush=True)
f32 = lambda L: np.array(L, dtype=np.float32).reshape(-1, 2)
FAILS = {"Incomplete", "Out", "Pass Offside", "Unknown"}
def around(p, O, T):
    """Context of the loss point p (attacking toward x = 120): the F_OLD features of the turnover model (as research/dq/fetch_v15.py)."""
    do = np.hypot(*(O - p).T); dt = np.hypot(*(T - p).T) if len(T) else np.array([99.0])
    return dict(n_opp3=int((do <= 3).sum()), n_opp6=int((do <= 6).sum()), nn_opp=float(min(do.min(), 30)), n_mate10=int((dt <= 10).sum()), nn_mate=float(min(dt.min(), 30)), n_opp_ahead=int((O[:, 0] < p[0]).sum()), n_cover=int((T[:, 0] < p[0]).sum()) if len(T) else 0)
dec = []; loss = []
for k, (comp, has360, m) in enumerate(jobs):
    mid = m["match_id"]; ev = get(f"events/{mid}.json")
    if ev is None: continue
    r = get(f"three-sixty/{mid}.json") if has360 else None; ff = {f["event_uuid"]: f for f in r} if r else {}
    shots = [(secs(e), e["team"]["id"], e["shot"]["statsbomb_xg"], e["possession"]) for e in ev if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] == "Open Play"]; n = len(ev)
    def outc(t0, tm, poss):
        w = lambda hz, same, poss_only: sum(x for tt, ti, x, pp in shots if (ti == tm) == same and t0 < tt <= t0 + hz and (not poss_only or pp == poss))
        return dict(own10=w(H, True, False), opp10=w(H, False, False), chain10=w(H, True, True), own20=w(H2, True, False), opp20=w(H2, False, False), chain20=w(H2, True, True))
    for i, e in enumerate(ev):
        t = e["type"]["name"]; loc = e.get("location")
        if loc is None: continue
        if has360 and e["id"] in ff:                                   # ---- losses of the ball anywhere (as fetch_v15.py)
            lp = lk = None
            if t == "Pass" and e["pass"].get("outcome", {}).get("name") in FAILS and "end_location" in e["pass"]: lp, lk = np.array(e["pass"]["end_location"][:2], float), "pass"
            elif t == "Dribble" and e["dribble"].get("outcome", {}).get("name") == "Incomplete": lp, lk = np.array(loc[:2], float), "dribble"
            elif t in ("Dispossessed", "Miscontrol"): lp, lk = np.array(loc[:2], float), t.lower()
            if lp is not None and lp[0] >= 0:
                fr = ff[e["id"]]["freeze_frame"]; Ol = np.array([q["location"][:2] for q in fr if not q["teammate"]], float).reshape(-1, 2); Tl = np.array([q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")], float).reshape(-1, 2)
                if len(Ol):
                    t0l = secs(e); loss.append(dict(kind=lk, comp=comp, match=mid, lx=float(lp[0]), ly=float(lp[1]), dist_own=float(np.hypot(lp[0], lp[1] - 40)), ang_own=float(abs(np.arctan2(44 - lp[1], lp[0]) - np.arctan2(36 - lp[1], lp[0]))), **around(lp, Ol, Tl), **outc(t0l, e["team"]["id"], e["possession"])))
        if t not in ("Shot", "Pass", "Carry"): continue
        a = np.array(loc[:2], float); t0 = secs(e); tm = e["team"]["id"]; row = None
        if t == "Shot":
            if loc[0] < 60 or e["shot"]["type"]["name"] != "Open Play" or e["shot"]["body_part"]["name"] == "Head": continue
            fr = e["shot"].get("freeze_frame", []); O = [q["location"][:2] for q in fr if not q["teammate"]]; Tm = [q["location"][:2] for q in fr if q["teammate"]]; gks = [q["location"][:2] for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
            if O: row = dict(kind="shot", tx=np.nan, ty=np.nan, sb_xg=e["shot"]["statsbomb_xg"], goal=int(e["shot"]["outcome"]["name"] == "Goal"), opps=f32(O), mates=f32(Tm), gk=np.array(gks[0], np.float32) if gks else None)
        elif has360 and e["id"] in ff:
            fr = ff[e["id"]]["freeze_frame"]; O = [q["location"][:2] for q in fr if not q["teammate"]]; Tm = [q["location"][:2] for q in fr if q["teammate"] and not q.get("actor")]; kp_ = [q["location"][:2] for q in fr if not q["teammate"] and q.get("keeper")]
            if not O: continue
            if t == "Pass":
                ps = e["pass"]
                if "type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" or ps.get("body_part", {}).get("name") == "Head" or "end_location" not in ps: continue
                row = dict(kind="pass", tx=ps["end_location"][0], ty=ps["end_location"][1], completed=int("outcome" not in ps), opps=f32(O), mates=f32(Tm), gk=np.array(kp_[0], np.float32) if kp_ else None)
            elif "end_location" in e["carry"] and float(np.hypot(*(np.array(e["carry"]["end_location"][:2]) - a))) >= 1.0:
                nxt = None
                for j in range(i + 1, min(i + 6, n)):
                    q = ev[j]
                    if q.get("player", {}).get("id") != e["player"]["id"] or q["type"]["name"] in ("Ball Receipt*", "Pressure", "Carry"): continue
                    nxt = q; break
                fail = int(nxt is not None and secs(nxt) - t0 < 6 and (nxt["type"]["name"] in ("Dispossessed", "Miscontrol") or (nxt["type"]["name"] == "Dribble" and nxt["dribble"].get("outcome", {}).get("name") == "Incomplete")))
                row = dict(kind="carry", tx=e["carry"]["end_location"][0], ty=e["carry"]["end_location"][1], fail=fail, opps=f32(O), mates=f32(Tm), gk=np.array(kp_[0], np.float32) if kp_ else None)
        if row is None: continue
        o = outc(t0, tm, e["possession"]); dec.append(dict(row, comp=comp, match=mid, period=e["period"], minute=e["minute"], second=e["second"], team=e["team"]["name"], index=e["index"], ax=float(a[0]), ay=float(a[1]), **o,
                                                           **({"cont10": sum(x for tt, ti, x, pp in shots if ti == tm and pp == e["possession"] and t0 < tt <= t0 + H), "cont20": sum(x for tt, ti, x, pp in shots if ti == tm and pp == e["possession"] and t0 < tt <= t0 + H2)} if row["kind"] == "shot" else {})))
    if k % 50 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, decisions {len(dec)}", flush=True)
Dd = pd.DataFrame(dec); Dd["test"] = Dd.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024"); Dd.to_pickle(f"{D}/v17_dec.pkl")
Lo = pd.DataFrame(loss); Lo["test"] = Lo.comp.str.contains("Euro 2024|Bundesliga 2023/2024|Major League Soccer 2023|Copa America 2024"); Lo.to_pickle(f"{D}/v17_loss.pkl"); print(f"losses {len(Lo)} {dict(Lo.kind.value_counts())}")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; decisions {len(Dd)}: {dict(Dd.kind.value_counts())}; test {int(Dd.test.sum())}; with keeper visible {Dd.gk.notna().mean()*100:.0f}%")
