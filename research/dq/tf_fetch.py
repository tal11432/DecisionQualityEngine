"""EU20 + WC22 events/360 in memory -> step table (deleted at the end of the task)."""
import json, os, gzip, time, urllib.request, numpy as np, pandas as pd, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); from tfcommon import steps, FEATS
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"; xfer = 0
def get(path):
    global xfer
    for att in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(BASE + path, headers={"Accept-Encoding": "gzip"}), timeout=60); b = r.read(); break
        except Exception as ex: print("retry", att, path, ex, flush=True); time.sleep(2 ** att)
    else: raise RuntimeError(path)
    xfer += len(b); return json.loads(gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b)
rows, pinfo = [], []
for comp, path in (("EU20", "55/43"), ("WC22", "43/106")):
    for k, m in enumerate(get(f"matches/{path}.json")):
        mid = m["match_id"]; ev = get(f"events/{mid}.json"); ff = {f["event_uuid"]: f for f in get(f"three-sixty/{mid}.json")}
        for e in ev:
            if e["type"]["name"] != "Pass" or e["id"] not in ff: continue
            ps = e["pass"]; oc = ps.get("outcome", {}).get("name")
            if "type" in ps or ps.get("height", {}).get("name") != "Ground Pass" or ps.get("cross") or ps.get("technique", {}).get("name") == "Through Ball" \
               or ps.get("body_part", {}).get("name") == "Head" or oc not in (None, "Incomplete"): continue
            a = np.array(e["location"][:2], float); end = np.array(ps["end_location"][:2], float)
            if np.hypot(*(end - a)) < 5: continue
            fr = ff[e["id"]]; opps = np.array([q["location"] for q in fr["freeze_frame"] if not q["teammate"]], float).reshape(-1, 2)
            R = steps(a, end, opps, oc == "Incomplete", fr.get("visible_area")); pid = len(pinfo)
            pinfo.append((pid, comp, mid, int(oc == "Incomplete"))); rows.append(np.column_stack([np.full(len(R), pid), R]))
        if k % 20 == 0: print(comp, k, f"{xfer/1e6:.0f} MB", flush=True)
S = pd.DataFrame(np.vstack(rows), columns=["pid"] + FEATS + ["event"]); Pi = pd.DataFrame(pinfo, columns=["pid", "comp", "match", "failed"])
S = S.merge(Pi[["pid", "comp", "match"]], on="pid"); S.to_pickle(f"{D}/steps_sb.pkl"); Pi.to_pickle(f"{D}/passes_sb.pkl")
print(f"DONE passes {len(Pi)} (failed {Pi.failed.sum()}), step rows {len(S)}, transferred {xfer/1e6:.0f} MB, nothing raw on disk")
