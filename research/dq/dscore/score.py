"""Decision Score v1.2 (PREREG_DQ_v13.txt, amendments E-H).
An ABSOLUTE 0-100 scale of the expected value of a decision. Every option of a moment (shot, pass to each visible teammate, the carry/take-on actually made, unforced turnover) gets its own score from its value Q = expected net xG of the possession in the next 10 s (xG for us minus xG for the opponent);
the decision gets the score of the option taken, the score of the best option and the gap.
Score = 100 * clip((Q + C) / (Q_hi + C), 0, 1): 100 at Q >= Q_hi = 0.70 (a move that very likely ends in a goal), 0 at Q <= -C (a loss that hands the opponent a counter-attack with a numerical advantage). Only Q enters: never the outcome, never other decisions (the percentile of amendment D is only the separate 'rarity' helper).
MARKED ASSUMPTION A1 (amendments G, H): the data cannot measure the cost of a counter-attack with a numerical advantage (a single 360 frame has no velocities; the tracking sample holds 2 such losses in 803). Where the visible balance
n_opp_ahead - n_cover is >= 2 at a loss point with x <= 70, the cost of losing the ball is set to max(L_model, C), C = 0.231. It is an assumption, not a measurement; the flag `counter_assumed` marks every option it touched and `assumption(False)` gives the model-only value."""
import os, sys, contextlib
import numpy as np, pandas as pd
DQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DQ not in sys.path: sys.path.insert(0, DQ)
from dscore import core
from dscore.core import GX, SC, Situation, sgm, ang, tri

_X = {}
def v13():
    if "m" not in _X: _X["m"] = pd.read_pickle(os.path.join(DQ, "v13_models.pkl"))
    return _X["m"]

def v15():
    """Turnover danger L (v13 feature set, refit on the v15 data) and the zero anchor C (amendments E and F; the counter features and the take-on model failed their gates and are not used)."""
    if "m15" not in _X: _X["m15"] = pd.read_pickle(os.path.join(DQ, "v15_models.pkl"))
    return _X["m15"]

def q_to_score(q, c=None, q_hi=None):
    """PRIMARY Decision Score: absolute linear map of the expected value Q (net xG of the next 10 s); 0 at Q = -C, 100 at Q = Q_hi."""
    c = v15()["C"] if c is None else c; q_hi = v13()["Q_hi"] if q_hi is None else q_hi
    return int(round(100 * min(1.0, max(0.0, (q + c) / (q_hi + c)))))

def set_reference(q_ref):
    """Reference distribution for the 'rarity' percentile: Q of the options TAKEN in a fixed sample of training decisions."""
    _X["ref"] = np.sort(np.asarray(q_ref, float))

def rank_score(q):
    """RARITY (not the Decision Score): percentile of Q among the reference decisions (amendment D, retired as the primary score by amendment E). Needs v14_reference.pkl built with the current models."""
    if "ref" not in _X:
        ref = pd.read_pickle(os.path.join(DQ, "v14_reference.pkl")); _X["ref"] = np.sort(np.asarray(ref["q"], float))
    r = _X["ref"]; lo, hi = np.searchsorted(r, q, "left"), np.searchsorted(r, q, "right"); return int(round(100 * (lo + hi) / 2 / len(r)))

def _around(p, O, T):
    """Context of point p (attacking toward x = 120), the same definitions as research/dq/fetch_v15.py: opponents near the ball and the counter numerical balance (opponents beyond the ball toward our goal against our players between the ball and our goal)."""
    do = np.hypot(*(O - p).T) if len(O) else np.array([99.0]); dt = np.hypot(*(T - p).T) if len(T) else np.array([99.0])
    ahead = O[O[:, 0] < p[0]] if len(O) else np.zeros((0, 2)); free = [o for o in ahead if not len(T) or np.hypot(*(T - o).T).min() > 6.0]
    return dict(n_opp2=int((do <= 2).sum()), n_opp3=int((do <= 3).sum()), n_opp6=int((do <= 6).sum()), nn_opp=float(min(do.min(), 30)), n_mate10=int((dt <= 10).sum()), nn_mate=float(min(dt.min(), 30)),
                n_opp_ahead=len(ahead), n_cover=int((T[:, 0] < p[0]).sum()) if len(T) else 0, n_free_run=len(free), run_x=float(ahead[:, 0].min()) if len(ahead) else 120.0,
                nn_run=float(min(np.hypot(*(np.array(free) - p).T).min(), 30)) if free else 30.0)

_ASSUME = {"on": True}
@contextlib.contextmanager
def assumption(on: bool):
    """Switch the marked assumption A1 on or off inside a with-block (default: on)."""
    old = _ASSUME["on"]; _ASSUME["on"] = bool(on)
    try: yield
    finally: _ASSUME["on"] = old

def counter_flag(p, O, T):
    """A1 trigger: visible numerical balance (opponents goal-side of the ball minus our players goal-side of it) >= 2 and the loss point at x <= 70 (StatsBomb units)."""
    p = np.asarray(p, float); c = _around(p, np.asarray(O, float).reshape(-1, 2), np.asarray(T, float).reshape(-1, 2)); return bool(c["n_opp_ahead"] - c["n_cover"] >= 2 and p[0] <= 70)

def loss_danger(p, O, T, kind="pass"):
    """L = expected opponent xG minus expected own xG in the 10 s after the ball is lost at p (units: xG); depends on the position and on who is around. The model cannot see a counter-attack with a numerical advantage (amendment F);
    under the marked assumption A1 (default on) a loss with a visible balance >= 2 at x <= 70 costs at least C."""
    m = v15()["L"]; c = _around(np.asarray(p, float), O, T); dist_own = float(np.hypot(p[0], p[1] - 40)); an = float(abs(np.arctan2(44 - p[1], p[0]) - np.arctan2(36 - p[1], p[0])))
    adv = c["n_opp_ahead"] - c["n_cover"]; adv_pos = float(min(max(adv, 0), 5))
    f = dict(c, dist_own=dist_own, ang_own=an, lx=p[0], d2=dist_own ** 2, ad=an * dist_own, k_pass=float(kind == "pass"), k_dribble=float(kind == "dribble"), k_dispossessed=float(kind == "dispossessed"), adv=float(adv), adv_pos=adv_pos, adv_x=adv_pos * (1 - p[0] / 120.0))
    X = [f[k] for k in m["F"]]; L = sgm(m["opp"], X) - sgm(m["own"], X)
    if _ASSUME["on"] and adv >= 2 and p[0] <= 70: L = max(L, float(v15()["C"]))
    return L

def carry_features(s: Situation, d, st, T, pdist):
    """Features of a carry from s.a to d (start context st = _around(s.a, ...)); the whole-pitch models of amendment J use dist ... pdist, s_* (no keeper distance); gk_d is kept for the frozen v13 models."""
    a, O, gk = s.a, s.O, s.gk; ar = _around(d, O, T); f = core.target_features(a, O, gk, d)
    f.update(n3=ar["n_opp3"], n6=ar["n_opp6"], nn=ar["nn_opp"], L=f["plen"], pdist=pdist, s_n_opp3=st["n_opp3"], s_nn_opp=st["nn_opp"], s_n_opp6=st["n_opp6"]); return f

def _carry_option(s: Situation, d, st, T, pdist, m):
    a, O, gk = s.a, s.O, s.gk; f = carry_features(s, d, st, T, pdist); M16 = core.models()["v16"]
    mk, mv = (M16["P_keep_all"], M16["V_pos_all"]) if M16 is not None else (m["P_keep"], m["V_pos_refit"])
    pk = sgm(mk, [f[k2] for k2 in mk["F"]]); vp = sgm(mv, [f[k2] for k2 in mv["F"]]); lo = loss_danger(d, O, T, "dispossessed")
    return dict(kind="carry", v=pk * vp - (1 - pk) * lo, p_keep=pk, value_after=vp, loss=lo, target=d, status="certain onside", counter_assumed=bool(_ASSUME["on"] and counter_flag(d, O, T)))

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
    T = np.array(s.mates, float).reshape(-1, 2); opts = []
    if s.a[0] >= core.SHOT_MIN_X:                                                    # amendment J: below x = 60 the shot option does not exist
        xg, cont, sv, _ = core.shot_value(s, header); opts.append(dict(kind="shot", v=sv, xg=xg, cont=cont, status="certain onside", target=None))
    for o in core.pass_options(s):
        lo = loss_danger(o["target"], s.O, T, "pass"); o = dict(o, kind="pass", loss=lo, v=o["xp"] * o["xa"] - (1 - o["xp"]) * lo, counter_assumed=bool(_ASSUME["on"] and counter_flag(o["target"], s.O, T))); opts.append(o)
    if carries: opts += carry_options(s)
    opts.append(dict(kind="turnover", v=-loss_danger(s.a, s.O, T, "pass"), status="certain onside", target=s.a, note="unforced turnover here", counter_assumed=bool(_ASSUME["on"] and counter_flag(s.a, s.O, T))))
    for o in opts: o["score"] = q_to_score(o["v"])
    return opts

def pass_value(s: Situation, end):
    """The ground pass actually made (target = where it was aimed or, for a failed pass, where the ball stopped): xP x chain value - (1 - xP) x L(target)."""
    T = np.array(s.mates, float).reshape(-1, 2); o = core.pass_option_at(s, end); lo = loss_danger(o["target"], s.O, T, "pass"); o.update(kind="pass", loss=lo, v=o["xp"] * o["xa"] - (1 - o["xp"]) * lo, counter_assumed=bool(_ASSUME["on"] and counter_flag(o["target"], s.O, T))); return o

def judge(s: Situation, chosen, header=0, include_doubt=False, scales=None):
    """chosen = ("shot",) | ("pass", xy) | ("carry", xy). Returns the option table, the score of the option taken, the best option's score and the gap (points and xG).
    The headline uses CERTAIN options only (offside doubt excluded); include_doubt=True also counts options in doubt."""
    opts = all_options(s, header); kind = chosen[0]
    if kind == "shot":
        ch = next((o for o in opts if o["kind"] == "shot"), None)
        if ch is None: raise ValueError("shot option does not exist below x = 60")
    elif kind == "carry": ch = carry_value(s, chosen[1])
    elif kind == "pass": ch = pass_value(s, chosen[1])
    else: raise ValueError(kind)
    if scales:                                   # perturbation: multiply the values of each kind of option (uncertainty band)
        for o in opts + ([] if any(o is ch for o in opts) else [ch]): o["v"] = o["v"] * scales.get(o["kind"], 1.0)
    ch["score"] = q_to_score(ch["v"])
    for o in opts: o["score"] = q_to_score(o["v"])
    ok = [o for o in opts if o["status"] == "certain onside" or (include_doubt and o["status"].startswith("DOUBT"))]
    best = max(ok + ([] if any(o is ch for o in ok) else [ch]), key=lambda o: o["v"])      # identity, not ==: the option dicts hold numpy arrays
    return dict(options=opts, chosen=ch, best=best, score_chosen=ch["score"], score_best=best["score"], gap_points=best["score"] - ch["score"], gap_xg=best["v"] - ch["v"],
                score_worst=min(o["score"] for o in opts), q_chosen=ch["v"], q_best=best["v"], counter_assumed_chosen=bool(ch.get("counter_assumed", False)), counter_assumed_best=bool(best.get("counter_assumed", False)),
                counter_assumed_any=any(o.get("counter_assumed", False) for o in opts))

def judge_both(s: Situation, chosen, header=0, include_doubt=False, scales=None):
    """The judgement under the marked assumption A1 (primary, as instructed) plus the model-only values: q_chosen_model, q_best_model, score_chosen_model, score_best_model, gap_xg_model, gap_points_model."""
    with assumption(True): j = judge(s, chosen, header, include_doubt, scales)
    with assumption(False): m = judge(s, chosen, header, include_doubt, scales)
    j.update(q_chosen_model=m["q_chosen"], q_best_model=m["q_best"], score_chosen_model=m["score_chosen"], score_best_model=m["score_best"], gap_xg_model=m["gap_xg"], gap_points_model=m["gap_points"]); return j
