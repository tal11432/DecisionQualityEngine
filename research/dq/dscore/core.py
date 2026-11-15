"""One home for the helpers that were copied between eval_v5 ... eval_v12b and research/cases/match_review.py:
geometry, the xP hazard, shot value, pass options with the offside-doubt classification, and the shot-vs-pass verdict.
Coordinates: StatsBomb units (120 x 80), attacking toward x = 120. Frozen models are read from research/dq/*.pkl."""
import os, sys
from dataclasses import dataclass, field
import numpy as np, pandas as pd
DQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DQ not in sys.path: sys.path.insert(0, DQ)
from tfcommon import steps, FEATS as HFT

GX = 120.0
SC = np.array([105 / 120, 68 / 80])              # StatsBomb units -> metres
FLOOR_SHORT, FLOOR_LONG = 0.03, 0.05             # DQ-v12b: absolute floors (xG) of the verdict rule; passes over 15 m use the long floor
LONG_PASS_M = 15.0

_M = {}
def models():
    """Models, loaded once: xG_geo (v5), shot continuation (v6), xP hazard (tf_model), and the whole-pitch models of amendment J (v16: xA_chain_all, P_keep_all, V_pos_all, per-zone xP recalibration) when v16_models.pkl exists."""
    if not _M:
        _M["v5"] = pd.read_pickle(os.path.join(DQ, "v5_models.pkl")); _M["v6"] = pd.read_pickle(os.path.join(DQ, "v6_models.pkl")); _M["hz"] = pd.read_pickle(os.path.join(DQ, "tf_model.pkl"))
        f16 = os.path.join(DQ, "v16_models.pkl"); _M["v16"] = pd.read_pickle(f16) if os.path.exists(f16) else None
    return _M

DEFAULT_GK = np.array([118.0, 40.0])             # the opposing keeper when he is not in the 360 frame (only the shot features use him)
SHOT_MIN_X = 60.0                                # below this the shot option does not exist (amendment J)
def zone(x): return "own third" if x < 40 else ("middle" if x < 80 else "attacking")

def sgm(m, X):
    """Logistic model stored as {mu, sd, b}: probability for one feature vector."""
    return float(1 / (1 + np.exp(-(np.concatenate([[1.0], (np.asarray(X, float) - m["mu"]) / m["sd"]]) @ m["b"]))))

def ang(x, y):
    """Goal-mouth angle (radians) seen from (x, y)."""
    return float(abs(np.arctan2(44 - y, GX - x) - np.arctan2(36 - y, GX - x)))

def tri(p, s):
    """Is point p inside the triangle shooter s - left post - right post?"""
    def f(ax, ay, bx, by, cx, cy): return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy)
    d1, d2, d3 = f(p[0], p[1], s[0], s[1], GX, 36), f(p[0], p[1], GX, 36, GX, 44), f(p[0], p[1], GX, 44, s[0], s[1])
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))

def in_box(p): return (p[0] >= 102) and (18 <= p[1] <= 62)

def xp_hazard(a, end, O):
    """Probability that a ground pass from a reaches end (interception hazard along the line; DQ-v2 model, not recalibrated)."""
    a, end = np.asarray(a, float), np.asarray(end, float)
    if float(np.hypot(*(end - a))) < 1e-6: return 1.0                    # zero-length pass: nothing to intercept
    HM = models()["hz"]; R = steps(a, end, np.asarray(O, float).reshape(-1, 2), False, None)
    l = np.column_stack([np.ones(len(R)), (R[:, :len(HFT)] - HM["mu"]) / HM["sd"]]) @ HM["b"]
    return float(np.exp(np.log1p(-1 / (1 + np.exp(-l))).sum()))

@dataclass
class Situation:
    """The ball carrier's moment: a = ball carrier, O = visible opponents (n, 2) incl. the keeper, mates = visible teammates (list of 2-vectors), gk = opposing keeper."""
    a: np.ndarray
    O: np.ndarray
    mates: list
    gk: np.ndarray
    def __post_init__(self):
        self.a = np.asarray(self.a, float); self.O = np.asarray(self.O, float).reshape(-1, 2); self.mates = [np.asarray(p, float) for p in self.mates]
        self.gk = DEFAULT_GK.copy() if self.gk is None or not np.all(np.isfinite(np.asarray(self.gk, float))) else np.asarray(self.gk, float)

def shot_features(s: Situation, header=0):
    a, O, gk = s.a, s.O, s.gk; dd = np.hypot(*(O - a).T); v = np.array([GX, 40.0]) - a; dist = float(np.hypot(*v)); an = ang(*a)
    return dict(dist=dist, angle=an, header=int(header), n_tri=sum(tri(o, a) for o in O), gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])),
                gk_lat=float(abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9)), n_opp2=int((dd <= 2).sum()), nn_opp=float(dd.min()), da=dist * an, ldist=float(np.log(dist)))

def shot_value(s: Situation, header=0):
    """(xG, continuation, total value, features): xG_geo + the expected later shots of the possession within 10 s."""
    M = models(); f = shot_features(s, header); xg = sgm(M["v5"]["xG_geo"], [f[k] for k in M["v5"]["xG_geo"]["F"]]); cont = sgm(M["v6"]["cont"], [f[k] for k in M["v6"]["cont"]["F"]])
    return xg, cont, xg + cont, f

def offside_status(margin_m, p_x, n_visible_opponents):
    """Certain onside needs a 1 m margin; a hidden defender can only make a player MORE onside, so 'offside by the visible players' is in doubt unless all 11 opponents are visible."""
    if p_x <= 60 or margin_m >= 1.0: return "certain onside"
    if margin_m > -1.0: return "DOUBT (within 1 m of the line)"
    if n_visible_opponents < 11: return "DOUBT (offside by the visible players, but a defender is out of view)"
    return "certain offside"

def offside_line(s: Situation):
    """x of the offside line (second-last visible opponent, or the ball if further)."""
    ox = np.sort(s.O[:, 0]); return max(ox[-2] if len(ox) > 1 else ox[-1], s.a[0])

def target_features(a, O, gk, p):
    """Features of the point p reached from the ball carrier a (a pass target or a carry destination): the v6 feature set plus ldist, used by the whole-pitch models of amendment J (no keeper distance in the whole-pitch version, but gk_d is still returned for the frozen v6 model)."""
    a, p = np.asarray(a, float), np.asarray(p, float); dp = np.hypot(*(O - p).T) if len(O) else np.array([99.0]); dist_p = float(np.hypot(GX - p[0], 40 - p[1])); an_p = ang(*p); pdist = float(np.hypot(GX - a[0], 40 - a[1]))
    return dict(dist=dist_p, angle=an_p, x=float(p[0]), d2=dist_p ** 2, ad=an_p * dist_p, ldist=float(np.log(max(dist_p, 1.0))), n_tri=sum(tri(o, p) for o in O), n3=int((dp <= 3).sum()), n6=int((dp <= 6).sum()), nn=float(dp.min()),
                gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), plen=float(np.hypot(*(p - a))), fwd=float(p[0] - a[0]), pdist=pdist)

def xp_recalibrated(xp, a_x):
    """xP with the per-zone logistic recalibration of amendment J when the fitted model recorded one for the passer's zone (else unchanged)."""
    m = models()["v16"]
    if m is None: return xp
    cal = m.get("xp_cal", {}).get(zone(a_x))
    if cal is None: return xp
    l = np.log(np.clip(xp, 1e-6, 1 - 1e-6) / (1 - np.clip(xp, 1e-6, 1 - 1e-6))); return float(1 / (1 + np.exp(-(cal[0] + cal[1] * l))))

def pass_option_at(s: Situation, p, line=None):
    """Ground pass from the ball carrier to point p (any distance): xP x chain value, offside margin and status. No range filter. Whole-pitch chain value (amendment J) when v16_models.pkl exists, else the frozen v6 model."""
    M = models(); a, O, gk = s.a, s.O, s.gk; p = np.asarray(p, float); line = offside_line(s) if line is None else line
    margin_m = float((line - p[0]) * 105 / 120); status = offside_status(margin_m, p[0], len(O)); d = float(np.hypot(*((p - a) * SC)))
    ft = target_features(a, O, gk, p); mx = M["v16"]["xA_chain_all"] if M["v16"] is not None else M["v6"]["xA_chain"]
    xa = sgm(mx, [ft[k] for k in mx["F"]]); xp = xp_recalibrated(xp_hazard(a, p, O), a[0])
    return dict(v=xp * xa, xp=xp, xa=xa, d=d, g=ft["dist"] * 105 / 120, margin=margin_m, status=status, target=p)

def pass_options(s: Situation):
    """Ground-pass options to every visible teammate 3-40 m away: value = xP x chain value, with offside status. Returns the list unsorted."""
    line = offside_line(s); return [pass_option_at(s, p, line) for p in s.mates if 3 <= float(np.hypot(*((p - s.a) * SC))) <= 40]

def rank_options(options):
    """(live, sure, gone): live = certain onside or in doubt, best value first; sure = certain onside only; gone = certainly offside."""
    live = sorted([c for c in options if c["status"] != "certain offside"], key=lambda c: -c["v"])
    return live, [c for c in live if c["status"] == "certain onside"], [c for c in options if c["status"] == "certain offside"]

def verdict(pass_value, shot_val, d=0.0):
    """Shot-vs-pass verdict: the ratio rule (pass value x0.7 / x1.3) AND a conservative absolute margin; 'leans' at half the floor."""
    floor = FLOOR_LONG if d > LONG_PASS_M else FLOOR_SHORT; lean = floor / 2; cp, cs = 0.7 * pass_value - shot_val, shot_val - 1.3 * pass_value
    return "pass clearly better" if cp >= floor else ("shot clearly better" if cs >= floor else ("leans pass" if cp >= lean else ("leans shot" if cs >= lean else "no clear difference")))
