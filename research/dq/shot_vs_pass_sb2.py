"""Shot vs pass on StatsBomb shot freeze frames (PREREG_SHOT_SB.txt). Streams events (gzip, in memory); writes a reduced table that is deleted after the run."""
import json, os, sys, gzip, time, urllib.request, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from tfcommon import steps, FEATS as HF
XG = pd.read_pickle(f"{D}/stage1b_models.pkl")["xG"]; HM = pd.read_pickle(f"{D}/tf_model.pkl"); GX = 120.0
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"; xfer = 0
def get(path):
    global xfer
    for att in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(BASE + path, headers={"Accept-Encoding": "gzip"}), timeout=60); b = r.read(); break
        except Exception as ex: print("retry", att, path, ex, flush=True); time.sleep(2 ** att)
    else: raise RuntimeError(path)
    xfer += len(b); return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
ang = lambda x, y: float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))
def intri(p, s):
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))
def xg(pos, opps, gk, extra=0):
    x = np.array([[np.hypot(GX - pos[0], 40 - pos[1]), ang(*pos), 0.0, sum(intri(o, pos) for o in opps) + extra, np.hypot(GX - gk[0], 40 - gk[1])]])
    return float(1 / (1 + np.exp(-(np.column_stack([np.ones(1), (x - XG["mu"]) / XG["sd"]]) @ XG["b"])))[0])
def xp(a, end, O):
    R = steps(a, end, O, False, None); lg = np.column_stack([np.ones(len(R)), (R[:, :len(HF)] - HM["mu"]) / HM["sd"]]) @ HM["b"]
    return float(np.exp(np.log1p(-1 / (1 + np.exp(-lg))).sum()))
m2 = lambda p: np.array([p[0] * 105 / 120, p[1] * 68 / 80])           # units -> metres (linear)
JOBS = {("Euro 2024", 55, 282), ("Copa America 2024", 223, 282), ("Bundesliga 2023/24", 9, 281), ("MLS 2023", 44, 107), ("AFCON 2023", 1267, 107), ("Ligue 1 2022/23", 7, 235), ("Ligue 1 2021/22", 7, 108), ("La Liga 2017/18", 11, 1), ("La Liga 2018/19", 11, 4), ("La Liga 2019/20", 11, 42), ("La Liga 2020/21", 11, 90), ("Champions League 2017/18", 16, 1), ("Champions League 2018/19", 16, 4), ("W Euro 2025", 53, 315), ("W Euro 2022", 53, 106), ("WWC 2023", 72, 107), ("WWC 2019", 72, 30), ("FA WSL 2023/24", 37, 281), ("Liga F 2023/24", 182, 281), ("Frauen Bundesliga 2023/24", 135, 281), ("Serie A Women 2023/24", 131, 281), ("NWSL 2023", 49, 107)}
jobs = [(n, m) for n, c, s in sorted(JOBS) for m in get(f"matches/{c}/{s}.json")]
shots, cands, excl = [], [], {"no_gk": 0}
for k, (comp, m) in enumerate(jobs):
    for e in get(f"events/{m['match_id']}.json"):
        if e["type"]["name"] != "Shot": continue
        s = e["shot"]
        if s["type"]["name"] != "Open Play" or s["body_part"]["name"] == "Head": continue
        fr = s.get("freeze_frame", []); a = np.array(e["location"][:2], float)
        opps = [np.array(q["location"][:2], float) for q in fr if not q["teammate"]]; mates = [np.array(q["location"][:2], float) for q in fr if q["teammate"]]
        gks = [np.array(q["location"][:2], float) for q in fr if not q["teammate"] and q.get("position", {}).get("name") == "Goalkeeper"]
        if not gks: excl["no_gk"] += 1; continue
        gk = gks[0]; O = np.array(opps).reshape(-1, 2); ox = np.sort(O[:, 0]); line = max(ox[-2] if len(ox) >= 2 else ox[-1], a[0])
        sid = len(shots); xs = xg(a, opps, gk)
        shots.append(dict(sid=sid, comp=comp, match=m["match_id"], date=m["match_date"], dist_m=float(np.hypot(*m2(np.array([GX, 40.0]) - a))), xg=xs, sb_xg=s["statsbomb_xg"],
                          goal=int(s["outcome"]["name"] == "Goal"), n_mates=len(mates), n_opps=len(opps)))
        for p in mates:
            d = float(np.hypot(*m2(p - a)))
            if not (3 <= d <= 40) or (p[0] > line and p[0] > 60): continue
            cands.append(dict(sid=sid, d_m=d, xP=xp(a, p, O), xA=xg(p, opps, gk), xA1=xg(p, opps, gk, extra=1)))
    if k % 50 == 0: print(k, comp, f"{xfer/1e6:.0f} MB, shots {len(shots)}", flush=True)
Sh, Cn = pd.DataFrame(shots), pd.DataFrame(cands); pd.to_pickle((Sh, Cn, excl), f"{D}/shot_vs_pass_sb2.pkl")
print(f"DONE matches {len(jobs)}, transferred {xfer/1e6:.0f} MB; shots {len(Sh)}, candidates {len(Cn)}, excluded {excl}")
