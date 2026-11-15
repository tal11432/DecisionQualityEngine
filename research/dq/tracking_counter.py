"""Amendment G of PREREG_DQ_v13.txt: how dangerous is a loss of the ball that gives the opponent a counter-attack with a numerical advantage, measured on tracking data WITH velocities
(Metrica Sample Games 1-3, IDSSE matches), and how well does the position-only proxy that the StatsBomb 360 data allows reproduce it.
Usage: python research/dq/tracking_counter.py [--counts-only] [match ...]   (default: metrica1 metrica2 metrica3 idsse:J03WPY)
No player names are written. Derived tables are not saved."""
import os, sys, glob, json
import numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(D)); sys.path.insert(0, ROOT); sys.path.insert(0, D)
FPS_WINDOW = 5                    # causal velocity window (frames), as in the existing pipeline (0.2 s at 25 Hz)
HORIZON = 10.0; NEXT_S = 6.0; HALF_L = 52.5

# ------------------------------------------------------------------ the indicator (pure function, rotated frame: team A attacks +x, A's goal at x = -52.5)
def counter_indicator(ball_x, xs, vxs, is_a, gk, carrier, speed=2.5, margin=5.0):
    """xs, vxs: x position and x velocity of every tracked player (NaN = not tracked); is_a: True for team A (the team that lost the ball); gk: True for goalkeepers; carrier: index of the ball carrier of A.
    B runners: B outfield players with x <= ball_x + margin and vx <= -speed. A back: A players (keeper included, carrier excluded) with x < ball_x. Returns (runners, back, adv)."""
    ok = np.isfinite(xs) & np.isfinite(vxs); idx = np.arange(len(xs))
    runners = int((ok & ~is_a & ~gk & (xs <= ball_x + margin) & (vxs <= -speed)).sum())
    back = int((np.isfinite(xs) & is_a & (idx != carrier) & (xs < ball_x)).sum())
    return runners, back, runners - back

def proxy_adv(ball_x, xs, is_a, carrier, keep):
    """The position-only analogue of the 360 frame: B players goal-side of the ball minus A players goal-side of the ball (carrier excluded), over the visible players only (keep: boolean mask)."""
    idx = np.arange(len(xs)); v = keep & np.isfinite(xs)
    return int((v & ~is_a & (xs < ball_x)).sum()) - int((v & is_a & (idx != carrier) & (xs < ball_x)).sum())

# ------------------------------------------------------------------ loading
class Match:
    def __init__(self, name, td, ev, kind):
        self.name, self.kind, self.td, self.ev = name, kind, td, ev.sort_values("t").reset_index(drop=True)
        fr = td.frames; self.t = fr.time_s.to_numpy(float); self.period = fr.period.to_numpy(int); cols = td.player_columns(); keys = sorted({c.rsplit("__", 1)[0] for c in cols})
        self.team = np.array([k.split("__")[0] for k in keys]); self.pid = [k.split("__")[1] for k in keys]
        self.X = np.column_stack([fr[f"{k}__x"].to_numpy(float) for k in keys]); self.Y = np.column_stack([fr[f"{k}__y"].to_numpy(float) for k in keys])
        self.bx = fr.ball_x.to_numpy(float); self.by = fr.ball_y.to_numpy(float); self.dir = {(tm, p): d for (tm, p), d in td.attack_dir.items()}
        self.gk = {}                                                     # per (team, period): the player whose median position is closest to its own goal
        for p in np.unique(self.period):
            sel = self.period == p
            for tm in ("Home", "Away"):
                d = self.dir.get((tm, int(p)))
                if d is None: continue
                cols_ = np.flatnonzero(self.team == tm); med = np.nanmedian(self.X[sel][:, cols_] * d, axis=0); self.gk[(tm, int(p))] = cols_[int(np.nanargmin(med))]
    def frame(self, t):
        k = int(np.searchsorted(self.t, t)); k = min(max(k, 0), len(self.t) - 1)
        if k > 0 and abs(self.t[k - 1] - t) < abs(self.t[k] - t): k -= 1
        return k
    def rotated(self, k, d):
        """positions at frame k rotated so that the direction with sign d attacks +x"""
        return self.X[k] * d, self.Y[k] * d, self.bx[k] * d, self.by[k] * d

def load(spec):
    from src.ingestion import metrica, epts, dfl
    if spec.startswith("metrica"):
        n = int(spec[-1]); b = f"{ROOT}/data/metrica/Sample_Game_{n}"
        if n in (1, 2):
            td = metrica.read_tracking(f"{b}/Sample_Game_{n}_RawTrackingData_Home_Team.csv", f"{b}/Sample_Game_{n}_RawTrackingData_Away_Team.csv"); ev = metrica.read_events(f"{b}/Sample_Game_{n}_RawEventsData.csv")
        else:
            td = epts.read_tracking(f"{b}/Sample_Game_3_tracking.txt", f"{b}/Sample_Game_3_metadata.xml"); ev = epts.read_events(f"{b}/Sample_Game_3_events.json", f"{b}/Sample_Game_3_metadata.xml")
        ev = ev.assign(t=ev.start_time_s.astype(float)); return Match(spec, td, ev, "metrica")
    jid = spec.split(":")[1]; b = f"{ROOT}/data/idsse/{jid}"; mi = glob.glob(f"{b}/*matchinformation*")[0]; meta = dfl.read_match_info(mi)
    td, qa, t0 = dfl.read_tracking(glob.glob(f"{b}/*positions*")[0], mi); bounds = {p: (g.time_s.min(), g.time_s.max()) for p, g in td.frames.groupby("period")}
    ev = dfl.read_events(glob.glob(f"{b}/*events_raw*")[0], meta, t0, bounds); ev = ev[~ev.is_delete].copy(); ev["t"] = ev.time_s.astype(float); m = Match(spec, td, ev, "idsse"); m.meta = meta; return m

# ------------------------------------------------------------------ losses
M_EXCL = {"OFFSIDE", "REFEREE HIT", "END HALF", "WOODWORK"}; M_ONBALL = {"RECOVERY", "PASS", "SHOT", "CHALLENGE"}; M_DEAD = {"BALL OUT", "SET PIECE", "FAULT RECEIVED", "CARD"}
D_DEAD = ("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff", "Foul", "Offside", "Caution", "RefereeBall", "FinalWhistle")
def _other(tm): return "Away" if tm == "Home" else "Home"

def losses(m: Match):
    ev = m.ev; out = []
    if m.kind == "metrica":
        ty = ev.type.to_numpy(); st = ev.subtype.fillna("").to_numpy(); tm = ev.team.to_numpy(); tt = ev.t.to_numpy(); n = len(ev)
        for i in range(n):
            if ty[i] != "BALL LOST" or st[i] in M_EXCL: continue
            a = tm[i]; ok = False
            for j in range(i + 1, n):
                if tt[j] - tt[i] > NEXT_S: break
                if ty[j] in M_DEAD: break
                if ty[j] in M_ONBALL: ok = (tm[j] == _other(a)); break
            if ok: out.append(dict(i=i, t=float(tt[i]), A=a, period=int(ev.period.iloc[i])))
    else:
        raw = ev.raw.apply(json.loads); ch = ev.chain.to_numpy(); tt = ev.t.to_numpy(); n = len(ev); side = m.meta.team_side
        def team_of(i):                                             # the team that performs the on-ball action
            if ch[i].startswith("TacklingGame"): return side.get(raw.iloc[i].get("TacklingGame.WinnerTeam"))
            return ev.team.iloc[i]
        for i in range(n):
            a = None
            if ch[i] in ("Play>Pass", "Play>Cross") and ev.evaluation.iloc[i] == "unsuccessful" and pd.isna(ev.set_piece.iloc[i]): a = ev.team.iloc[i]
            elif ch[i] == "TacklingGame":
                r = raw.iloc[i]
                if r.get("TacklingGame.LoserRole") == "withBallControl" and r.get("TacklingGame.PossessionChange") == "true": a = side.get(r.get("TacklingGame.LoserTeam"))
            if a not in ("Home", "Away") or not np.isfinite(ev.period.iloc[i]): continue
            ok = False
            for j in range(i + 1, n):
                if tt[j] - tt[i] > NEXT_S: break
                if ch[j].startswith(D_DEAD): break
                if ch[j].startswith(("Play", "TacklingGame", "BallClaiming", "OtherBallAction")): ok = (team_of(j) == _other(a)); break
            if ok: out.append(dict(i=i, t=float(tt[i]), A=a, period=int(ev.period.iloc[i])))
    return out

# ------------------------------------------------------------------ shots and xG
def shots(m: Match):
    from dscore.core import Situation, shot_value
    ev = m.ev; res = []
    if m.kind == "metrica": sel = ev[ev.type == "SHOT"]; lag = 0.0
    else: sel = ev[ev.chain.str.startswith("ShotAtGoal")]; lag = 0.2
    for r in sel.itertuples():
        tm = r.team; p = int(r.period) if np.isfinite(r.period) else None
        if tm not in ("Home", "Away") or p is None or (tm, p) not in m.dir: continue
        d = m.dir[(tm, p)]; k = m.frame(r.t - lag); xs, ys, bx, by = m.rotated(k, d)
        sx, sy = (r.start_x, r.start_y) if m.kind == "metrica" else (r.x, r.y)
        if not (np.isfinite(sx) and np.isfinite(sy)): continue
        sx, sy = sx * d, sy * d; own = m.team == tm; vis = np.isfinite(xs) & np.isfinite(ys)
        O = np.column_stack([xs[~own & vis], ys[~own & vis]]); T = np.column_stack([xs[own & vis], ys[own & vis]])
        if len(O) == 0: continue
        conv = lambda x, y: np.array([(x + HALF_L) * 120 / 105, (y + 34) * 80 / 68])
        a = conv(sx, sy); gk = conv(*O[np.argmin(np.hypot(O[:, 0] - HALF_L, O[:, 1]))]); sit = Situation(a, np.array([conv(*o) for o in O]), [conv(*t_) for t_ in T], gk)
        header = int(m.kind == "metrica" and isinstance(r.subtype, str) and r.subtype.startswith("HEAD"))
        res.append(dict(t=float(r.t), team=tm, xg=float(shot_value(sit, header)[0])))
    return pd.DataFrame(res, columns=["t", "team", "xg"])

# ------------------------------------------------------------------ features per loss
def features(m: Match, L, speed=2.5, margin=5.0, rng=None, draws=0):
    rows = []
    for l in L:
        A, B = l["A"], _other(l["A"]); d = m.dir.get((A, l["period"]))
        if d is None: continue
        k = m.frame(l["t"]); k0 = k - FPS_WINDOW
        if k0 < 0 or m.period[k0] != m.period[k] or abs(m.t[k] - l["t"]) > 0.5: continue
        xs, ys, bx, by = m.rotated(k, d); xs0, ys0, _, _ = m.rotated(k0, d); dt = m.t[k] - m.t[k0]; vx = (xs - xs0) / dt
        if not (np.isfinite(bx) and np.isfinite(by)): continue
        is_a = m.team == A; gk = np.zeros(len(xs), bool)
        for tm in ("Home", "Away"):
            g = m.gk.get((tm, l["period"]))
            if g is not None: gk[g] = True
        dist = np.where(is_a & np.isfinite(xs), np.hypot(xs - bx, ys - by), np.inf); carrier = int(np.argmin(dist))
        if not np.isfinite(dist[carrier]) or dist[carrier] > 6.0: carrier = -1       # carrier not found near the ball (event position error): nobody excluded
        run, back, adv = counter_indicator(bx, xs, vx, is_a, gk, carrier, speed, margin)
        row = dict(match=m.name, period=l["period"], t=l["t"], A=A, ball_x=bx, runners=run, back=back, adv=adv, x_sb=(bx + HALF_L) * 120 / 105)
        if draws:
            px = []
            for _ in range(draws): keep = rng.random(len(xs)) < 0.65; px.append(proxy_adv(bx, xs, is_a, carrier, keep))
            row["proxy"] = np.array(px)
        rows.append(row)
    return rows

def outcomes(rows, S):
    for r in rows:
        s = S[(S.t > r["t"]) & (S.t <= r["t"] + HORIZON)]; B = _other(r["A"])
        r["opp"] = float(s[s.team == B].xg.sum()); r["own"] = float(s[s.team == r["A"]].xg.sum()); r["net"] = r["opp"] - r["own"]
    return rows

# ------------------------------------------------------------------ statistics
rng = np.random.default_rng(0)
def boot(df, f, B=2000):
    blocks = df.groupby(["match", "period"]).indices; keys = list(blocks); vals = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys)); idx = np.concatenate([blocks[keys[i]] for i in pick]); vals.append(f(df.iloc[idx]))
    v = np.array(vals); v = v[np.isfinite(v)]; return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
def mean_net(df, cond): g = df[cond(df)]; return g.net.mean() if len(g) else np.nan

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; counts_only = "--counts-only" in sys.argv
    specs = args or ["metrica1", "metrica2", "metrica3", "idsse:J03WPY"]; allrows = []; kept = []
    for sp in specs:
        m = load(sp); L = losses(m); rows = features(m, L, rng=rng, draws=0 if counts_only else 50)
        print(f"{sp}: {len(m.t)} frames, {len(L)} losses with a next-ball-event of the opponent within {NEXT_S:.0f} s, {len(rows)} with usable frames; adv>=2: {sum(r['adv']>=2 for r in rows)}, adv==1: {sum(r['adv']==1 for r in rows)}, adv<=0: {sum(r['adv']<=0 for r in rows)}", flush=True)
        if not counts_only: S = shots(m); print(f"   shots {len(S)}, xG sum {S.xg.sum():.2f} (mean {S.xg.mean():.3f})"); rows = outcomes(rows, S)
        allrows += rows; kept.append((m, L, S))
    if counts_only:
        print(f"\nOUTCOME-FREE COUNTS: losses {len(allrows)}; adv>=2: {sum(r['adv']>=2 for r in allrows)}; adv==1: {sum(r['adv']==1 for r in allrows)}; adv<=0: {sum(r['adv']<=0 for r in allrows)}; G-T1 (primary class >= 100): {'ok' if sum(r['adv']>=2 for r in allrows) >= 100 else 'NOT YET'}"); sys.exit(0)
    df = pd.DataFrame(allrows); prox = np.stack([r["proxy"] for r in allrows]); df = df.drop(columns=["proxy"])
    print(f"\n== losses {len(df)} in {df.groupby(['match','period']).ngroups} match-period blocks; realised net danger (opponent xG - own xG within {HORIZON:.0f} s): overall mean {df.net.mean():+.4f}; share with an opponent shot {(df.opp>0).mean()*100:.1f}%")
    cls = {"adv>=2 (primary)": df.adv >= 2, "adv==1": df.adv == 1, "adv<=0 (reference)": df.adv <= 0}
    for lab, c in cls.items():
        g = df[c]; lo, hi = boot(df, lambda x, c=c: x[(x.adv >= 2) if lab.startswith("adv>=2") else ((x.adv == 1) if lab == "adv==1" else (x.adv <= 0))].net.mean())
        print(f"   {lab}: n={len(g)}, mean net danger {g.net.mean():+.4f} [{lo:+.4f},{hi:+.4f}], opponent xG {g.opp.mean():.4f}, own xG {g.own.mean():.4f}, share with an opponent shot {(g.opp>0).mean()*100:.1f}%")
    P, R_ = df[df.adv >= 2], df[df.adv <= 0]; diff = P.net.mean() - R_.net.mean(); lo, hi = boot(df, lambda x: x[x.adv >= 2].net.mean() - x[x.adv <= 0].net.mean())
    n2 = len(P); half = (boot(df, lambda x: x[x.adv >= 2].net.mean())[1] - boot(df, lambda x: x[x.adv >= 2].net.mean())[0]) / 2; cm = P.net.mean(); clo, chi = boot(df, lambda x: x[x.adv >= 2].net.mean())
    g1 = n2 >= 100; g2 = g1 and lo > 0; g3 = g1 and g2 and half <= 0.05 and 0.05 <= cm <= 0.35
    print(f"\nG-T1 identifiability: n(adv>=2) = {n2} (need >= 100; < 60 after all matches = not supported) => {'PASS' if g1 else 'FAIL'}")
    print(f"G-T2 discrimination: net danger adv>=2 minus adv<=0 = {diff:+.4f} [{lo:+.4f},{hi:+.4f}] => " + ("PASS" if g2 else ("FAIL" if g1 else f"NOT EVALUATED: the primary class has {n2} losses in {P.groupby(['match','period']).ngroups} blocks, the bootstrap interval is degenerate and means nothing")))
    print(f"G-T3 anchor: C_meas = {cm:.4f} [{clo:.4f},{chi:.4f}] (half-width {half:.4f}; needs n>=100, G-T2, half-width <= 0.05 and C_meas in [0.05, 0.35]) => {'PASS: use C_meas' if g3 else 'FAIL: C stays 0.231 (and with n = ' + str(n2) + ' the value is not an estimate)'}")
    print("   by zone of the loss (StatsBomb x of the ball): " + "; ".join(f"{b}: n={len(g)}, mean net {g.net.mean():+.4f}" for b, g in df[df.adv >= 2].groupby(pd.cut(df[df.adv >= 2].x_sb, [-1, 40, 60, 80, 121]), observed=True)))
    # ---- proxy check (what the 360 data can see): losses at x <= 70, proxy adv >= 2 from a random 65% of the players
    z = (df.x_sb <= 70).to_numpy(); vel = (df.adv >= 2).to_numpy(); netv = df.net.to_numpy(); prec, rec, nflag, mflag, mrest = [], [], [], [], []
    for j in range(prox.shape[1]):
        f = (prox[:, j] >= 2) & z; nflag.append(f.sum()); prec.append((vel & f).sum() / max(f.sum(), 1)); rec.append((vel & f).sum() / max((vel & z).sum(), 1)); mflag.append(netv[f].mean() if f.any() else np.nan); mrest.append(netv[z & ~f].mean())
    print(f"\n== PROXY (visible-position balance, no velocities, random 65% of the players, 50 draws; losses at StatsBomb x <= 70, n={int(z.sum())}): flagged per draw {np.mean(nflag):.1f}; precision for the velocity class {np.nanmean(prec):.3f}, recall {np.nanmean(rec):.3f}; mean net danger of flagged losses {np.nanmean(mflag):+.4f} vs not flagged {np.nanmean(mrest):+.4f}")
    print(f"   velocity class among losses at x <= 70: n={int((vel & z).sum())}, mean net danger {netv[vel & z].mean():+.4f}; base rate of the velocity class {np.mean(vel[z])*100:.1f}%")
    # ---- sensitivity (descriptive only; not used for any decision)
    print("\n== SENSITIVITY (descriptive only, not used for any decision): the primary class (adv >= 2) and the secondary class (adv >= 1) under other thresholds")
    for sp_, mg in ((2.5, 5.0), (2.0, 5.0), (3.0, 5.0), (2.5, 3.0), (2.5, 7.0)):
        rr = []
        for m, L, S in kept: rr += outcomes(features(m, L, speed=sp_, margin=mg), S)
        q = pd.DataFrame(rr); a2, a1 = q[q.adv >= 2], q[q.adv >= 1]
        print(f"   speed {sp_} m/s, margin {mg} m: adv>=2 n={len(a2)}, mean net {a2.net.mean() if len(a2) else float('nan'):+.4f}; adv>=1 n={len(a1)}, mean net {a1.net.mean() if len(a1) else float('nan'):+.4f}; reference n={int((q.adv<=0).sum())}, mean net {q[q.adv<=0].net.mean():+.4f}")
    # ---- continuous view (descriptive): net danger by adv bin over ALL losses
    print("   net danger by adv bin (all losses): " + "; ".join(f"{b}: n={len(g)}, mean {g.net.mean():+.4f}" for b, g in df.groupby(pd.cut(df.adv, [-99, -6, -3, -1, 0, 99]), observed=True)))
