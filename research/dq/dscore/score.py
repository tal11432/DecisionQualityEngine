"""Decision Score v1 (PREREG_DQ_v13.txt incl. amendments A and B).
An ABSOLUTE 0-100 scale over general decisions: every option of a moment (shot, pass to each visible teammate, carries, unforced turnover) gets its own score from its value Q (expected net xG of the possession in the next 10 s);
the decision gets the score of the option taken, the score of the best option and the gap. 50 = changes nothing in expectation; 100 = Q >= Q_hi (0.70); 0 = Q <= -Q_lo (an opponent left one-on-one, Q_lo = mean one-on-one xG)."""
import os, sys
import numpy as np, pandas as pd
DQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DQ not in sys.path: sys.path.insert(0, DQ)
from dscore import core
from dscore.core import GX, SC, Situation, sgm, ang, tri

_X = {}
def v13():
    if not _X: _X["m"] = pd.read_pickle(os.path.join(DQ, "v13_models.pkl"))
    return _X["m"]

def q_to_score(q, q_lo=None, q_hi=None):
    """Piecewise-linear map of Q to 0-100 with 0 -> 50 (an action that changes nothing)."""
    m = v13(); q_lo = m["Q_lo"] if q_lo is None else q_lo; q_hi = m["Q_hi"] if q_hi is None else q_hi
    s = 50 + 50 * min(q, q_hi) / q_hi if q >= 0 else 50 - 50 * min(-q, q_lo) / q_lo
    return int(round(s))

def _around(p, O, T):
    do = np.hypot(*(O - p).T) if len(O) else np.array([99.0]); dt = np.hypot(*(T - p).T) if len(T) else np.array([99.0])
    return dict(n_opp2=int((do <= 2).sum()), n_opp3=int((do <= 3).sum()), n_opp6=int((do <= 6).sum()), nn_opp=float(min(do.min(), 30)), n_mate10=int((dt <= 10).sum()), nn_mate=float(min(dt.min(), 30)),
                n_opp_ahead=int((O[:, 0] < p[0]).sum()) if len(O) else 0, n_cover=int((T[:, 0] < p[0]).sum()) if len(T) else 0)

def loss_danger(p, O, T, kind="pass"):
    """L = expected opponent xG minus expected own xG in the 10 s after the ball is lost at p (units: xG); situation dependent."""
    m = v13()["L"]; c = _around(np.asarray(p, float), O, T); dist_own = float(np.hypot(p[0], p[1] - 40)); an = float(abs(np.arctan2(44 - p[1], p[0]) - np.arctan2(36 - p[1], p[0])))
    f = dict(c, dist_own=dist_own, ang_own=an, lx=p[0], d2=dist_own ** 2, ad=an * dist_own, k_pass=float(kind == "pass"), k_dribble=0.0, k_dispossessed=float(kind == "dispossessed"))
    X = [f[k] for k in m["F"]]; return sgm(m["opp"], X) - sgm(m["own"], X)

def _carry_option(s: Situation, d, st, T, pdist, m):
    a, O, gk = s.a, s.O, s.gk; ar = _around(d, O, T); dist = float(np.hypot(GX - d[0], 40 - d[1])); an = ang(*d); Lu = float(np.hypot(*(d - a)))
    f = dict(dist=dist, angle=an, x=d[0], d2=dist ** 2, ad=an * dist, n_tri=sum(tri(o, d) for o in O), n3=ar["n_opp3"], n6=ar["n_opp6"], nn=ar["nn_opp"], gk_d=float(np.hypot(GX - gk[0], 40 - gk[1])), plen=Lu, L=Lu, fwd=float(d[0] - a[0]), pdist=pdist,
             s_n_opp3=st["n_opp3"], s_nn_opp=st["nn_opp"], s_n_opp6=st["n_opp6"])
    pk = sgm(m["P_keep"], [f[k2] for k2 in m["P_keep"]["F"]]); vp = sgm(m["V_pos_refit"], [f[k2] for k2 in m["V_pos_refit"]["F"]]); lo = loss_danger(d, O, T, "dispossessed")
    return dict(kind="carry", v=pk * vp - (1 - pk) * lo, p_keep=pk, value_after=vp, loss=lo, target=d, status="certain onside")

def carry_options(s: Situation, lengths_m=(3, 6, 10), n_dirs=8):
    """Hypothetical carries (3/6/10 m, 8 directions). NOT supported as alternatives in v1 (PREREG_DQ_v13 amendment C): V_pos is extrapolated and the maximum over destinations is biased up."""
    m = v13(); a = s.a; T = np.array(s.mates, float).reshape(-1, 2); st = _around(a, s.O, T); pdist = float(np.hypot(GX - a[0], 40 - a[1])); out = []
    for Lm in lengths_m:
        for k in range(n_dirs):
            th = 2 * np.pi * k / n_dirs; d = a + np.array([np.cos(th) * Lm / SC[0], np.sin(th) * Lm / SC[1]])
            if 0 <= d[0] <= GX and 0 <= d[1] <= 80: o = _carry_option(s, d, st, T, pdist, m); o.update(length_m=Lm, direction_deg=int(round(np.degrees(th)))); out.append(o)
    return out

def carry_value(s: Situation, dest):
    """The carry actually made (destination dest): P_keep x V_pos(dest) - (1 - P_keep) x L(dest), calibrated on carries that were made."""
    m = v13(); T = np.array(s.mates, float).reshape(-1, 2)
    return _carry_option(s, np.asarray(dest, float), _around(s.a, s.O, T), T, float(np.hypot(GX - s.a[0], 40 - s.a[1])), m)

def all_options(s: Situation, header=0, carries=False):
    """Every option of the moment with its value Q: shot, passes (with offside status), the unforced turnover; hypothetical carries only when carries=True (NOT supported in v1, amendment C)."""
    T = np.array(s.mates, float).reshape(-1, 2); xg, cont, sv, _ = core.shot_value(s, header); opts = [dict(kind="shot", v=sv, xg=xg, cont=cont, status="certain onside", target=None)]
    for o in core.pass_options(s):
        lo = loss_danger(o["target"], s.O, T, "pass"); o = dict(o, kind="pass", loss=lo, v=o["xp"] * o["xa"] - (1 - o["xp"]) * lo); opts.append(o)
    if carries: opts += carry_options(s)
    opts.append(dict(kind="turnover", v=-loss_danger(s.a, s.O, T, "pass"), status="certain onside", target=s.a, note="unforced turnover here"))
    for o in opts: o["score"] = q_to_score(o["v"])
    return opts

def pass_value(s: Situation, end):
    """The ground pass actually made (target = where it was aimed or, for a failed pass, where the ball stopped): xP x chain value - (1 - xP) x L(target)."""
    T = np.array(s.mates, float).reshape(-1, 2); o = core.pass_option_at(s, end); lo = loss_danger(o["target"], s.O, T, "pass"); o.update(kind="pass", loss=lo, v=o["xp"] * o["xa"] - (1 - o["xp"]) * lo); return o

def judge(s: Situation, chosen, header=0, include_doubt=False, scales=None):
    """chosen = ("shot",) | ("pass", xy) | ("carry", xy). Returns the option table, the score of the option taken, the best option's score and the gap (points and xG).
    The headline uses CERTAIN options only (offside doubt excluded); include_doubt=True also counts options in doubt."""
    opts = all_options(s, header); kind = chosen[0]
    if kind == "shot": ch = next(o for o in opts if o["kind"] == "shot")
    elif kind == "carry": ch = carry_value(s, chosen[1])
    elif kind == "pass": ch = pass_value(s, chosen[1])
    else: raise ValueError(kind)
    if scales:                                   # perturbation: multiply the values of each kind of option (uncertainty band)
        for o in opts + ([] if any(o is ch for o in opts) else [ch]): o["v"] = o["v"] * scales.get(o["kind"], 1.0)
    ch["score"] = q_to_score(ch["v"])
    for o in opts: o["score"] = q_to_score(o["v"])
    ok = [o for o in opts if o["status"] == "certain onside" or (include_doubt and o["status"].startswith("DOUBT"))]
    best = max(ok + ([ch] if ch not in ok else []), key=lambda o: o["v"])
    return dict(options=opts, chosen=ch, best=best, score_chosen=ch["score"], score_best=best["score"], gap_points=best["score"] - ch["score"], gap_xg=best["v"] - ch["v"],
                score_worst=min(o["score"] for o in opts), q_chosen=ch["v"], q_best=best["v"])
