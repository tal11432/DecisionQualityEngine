"""H1 (passer-motion direction): analysis frozen in PREREGISTRATION_H1.txt. Scratch only."""
import sys; sys.path.insert(0, ".")
H = "research/choice/"
sys.argv = ["", H + "cands.pkl", H + "passes_cov.pkl", H + "unused.pkl"]
import numpy as np, pandas as pd
import model as mdl
from src.config import load_config
from src.cli import _load
from src.pass_completion_features import _Track
from src.pass_dataset import DecisionContract
from src.processing.sync import synchronize_events
from src.pass_completion import Prep
cfg = load_config(); tol = cfg["sync"]["tolerance_s"]; Wn = DecisionContract.from_config(cfg).velocity_window_frames
Cd, Pd0 = mdl.Cd, pd.read_pickle(H + "passes_cov.pkl")

def vel(Xj, Yj, tm, r0, lag):          # causal backward difference over Wn frames, ending `lag` frames before release
    a, b = r0 - lag, r0 - lag - Wn
    dt = float(tm[a] - tm[b]); return np.array([Xj[a] - Xj[b], Yj[a] - Yj[b]]) / dt

# ---- per-pass oriented velocity (current, lagged); P4 leakage check on the fly
vcur, vlag, leak_bad = {}, {}, 0
for g in (1, 2, 3):
    td, ev = _load(cfg, g); T = _Track.of(td); sub = Pd0[Pd0.game == g]
    sync = synchronize_events(ev.loc[sub.idx, ["team", "period", "start_time_s", "from_player"]], td.frames, tol, "start_time_s", None).matched.set_index("event_idx")["frame_row"]
    for _, r in sub.iterrows():
        r0 = int(sync[r.idx]); j = next(k for k in range(len(T.pid)) if T.team[k] == r.team and T.pid[k] == r.passer); sg = T.sign(r.team, r0)
        v1 = vel(T.X[:, j], T.Y[:, j], T.time, r0, 0) * sg; v2 = vel(T.X[:, j], T.Y[:, j], T.time, r0, Wn) * sg
        Xm, Ym = T.X[:, j].copy(), T.Y[:, j].copy(); Xm[r0 + 1:] = np.nan; Ym[r0 + 1:] = np.nan   # P4: remove everything after release
        if not np.allclose(vel(Xm, Ym, T.time, r0, 0) * sg, v1, equal_nan=True): leak_bad += 1
        vcur[r["pass"]] = v1; vlag[r["pass"]] = v2
print(f"P4 leakage check: passes where F1 inputs changed after NaN-ing all frames > release: {leak_bad} of {len(vcur)}")

pos = {p: i for i, p in enumerate(Pd0["pass"])}; pi = Cd["pass"].map(pos).to_numpy()
VC = np.array([vcur[p] for p in Pd0["pass"]]); VL = np.array([vlag[p] for p in Pd0["pass"]])
PX, PY = Pd0.px.to_numpy(), Pd0.py.to_numpy()
vec = np.column_stack([Cd.x_att.to_numpy() - PX[pi], Cd.y_att.to_numpy() - PY[pi]]); vn = np.hypot(*vec.T) + 1e-9
def feat(V, thr, order=None):
    V = V if order is None else V[order]; sp = np.hypot(*V.T)[pi]; cosv = (V[pi] * vec).sum(1) / (sp * vn + 1e-9)
    return np.where(sp >= thr, cosv, 0.0)
sp_cur = np.hypot(*VC.T)
# consistency with the earlier post-hoc feature (Cd.cos_vel, Pd.speed)
chk = feat(VC, 1.0); old = np.where(Cd["pass"].map(Pd0.set_index("pass").speed) >= 1.0, Cd.cos_vel, 0.0)
print(f"max |F1 - earlier post-hoc feature| = {np.abs(chk - old).max():.2e}")
Cd["F1"], Cd["F05"], Cd["F2"], Cd["FL"] = feat(VC, 1.0), feat(VC, 0.5), feat(VC, 2.0), feat(VL, 1.0)
Cd["ci"] = np.arange(len(Cd))
BASE = ["dist", "forward", "lateral", "rec_press", "margin"]; rng = np.random.default_rng(0)

def run(rule, extra):
    mdl.FEATS = BASE + extra; out = {}
    for te in (1, 2, 3): out[te] = mdl.evaluate(rule, [g for g in (1, 2, 3) if g != te], te)
    return out
def pooled(o, key): return np.concatenate([o[g][key] for g in (1, 2, 3)])
def bci(a_by_game, nb=1000):
    out = {}
    for g in a_by_game:
        a = a_by_game[g]; idx = rng.integers(0, len(a), (nb, len(a))); b = a[idx].mean(1); out[g] = (a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5))
    return out
def paired(o1, o0, key, label):
    res = {}
    for g in (1, 2, 3):
        d = (o1[g][key] - o0[g][key]); idx = rng.integers(0, len(d), (1000, len(d))); b = d[idx].mean(1); res[g] = (d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5))
    d = pooled(o1, key) - pooled(o0, key); idx = rng.integers(0, len(d), (1000, len(d))); b = d[idx].mean(1); res["all"] = (d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5))
    print(f"  {label:<34} " + "  ".join(f"{k}: {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]" for k, v in res.items())); return res

print("\n=== PRIMARY (R0): M1 = M0 + F1 vs M0 ===")
M0 = run("R0_primary", []); M1 = run("R0_primary", ["F1"])
for g in (1, 2, 3): print(f"  G{g}: n={M0[g]['n']}  ridge M0 {M0[g]['lam']} M1 {M1[g]['lam']}  top-1 M0 {M0[g]['model_hit'].mean():.3f} M1 {M1[g]['model_hit'].mean():.3f}  logloss M0 {M0[g]['model_ll'].mean():.3f} M1 {M1[g]['model_ll'].mean():.3f}  MRR {M0[g]['model_rr'].mean():.3f}->{M1[g]['model_rr'].mean():.3f}")
pl = paired(M1, M0, "model_ll", "P1 paired log loss M1-M0"); pt = paired(M1, M0, "model_hit", "P2 paired top-1  M1-M0"); paired(M1, M0, "model_rr", "   paired MRR   M1-M0")
P1 = all(pl[g][0] < 0 for g in (1, 2, 3)) and pl["all"][2] < 0; P2 = all(pt[g][0] > 0 for g in (1, 2, 3)) and pt["all"][1] > 0
print(f"  P1 {'PASS' if P1 else 'FAIL'}   P2 {'PASS' if P2 else 'FAIL'}   F1 coefficient (standardised, per held-out fold): " + ", ".join(f"G{g} {M1[g]['beta'][-1]:+.2f}" for g in (1, 2, 3)))

# ---- P3 placebo (ridge fixed 1.0, pooled held-out log-loss gain)
folds = {}
for te in (1, 2, 3):
    tr = [g for g in (1, 2, 3) if g != te]; dtr, st = mdl.sets_for("R0_primary", tr); dte, se = mdl.sets_for("R0_primary", [te]); folds[te] = (dtr, st, dte, se)
def gain(Fcol):
    tot0 = tot1 = n = 0
    for te, (dtr, st, dte, se) in folds.items():
        for ex, acc in (([], 0), (["F"], 1)):
            cols = BASE + ex; A = dtr[BASE].copy(); B = dte[BASE].copy()
            if ex: A["F"] = Fcol[dtr.ci.to_numpy()]; B["F"] = Fcol[dte.ci.to_numpy()]
            pr = Prep().fit(A[cols].to_numpy(float)); b = mdl.fit_clogit(pr.transform(A[cols].to_numpy(float)), mdl.chosen_idx(dtr, st), st, 1.0)
            p = mdl.probs(pr.transform(B[cols].to_numpy(float)), b, se); ll = -np.log(p[mdl.chosen_idx(dte, se)]).sum()
            if acc: tot1 += ll
            else: tot0 += ll
        n += len(se)
    return (tot1 - tot0) / n
real = gain(Cd["F1"].to_numpy()); games = Pd0.game.to_numpy(); null = []
for _ in range(200):
    order = np.arange(len(Pd0))
    for g in (1, 2, 3): ix = np.flatnonzero(games == g); order[ix] = rng.permutation(ix)
    null.append(gain(feat(VC, 1.0, order)))
null = np.array(null); pv = (null <= real).mean()   # gain is a log-loss difference: more negative = better
print(f"\n=== P3 placebo (ridge 1.0; pooled held-out log-loss difference M1-M0, negative = better) ===\n  real {real:+.4f}; placebo mean {null.mean():+.4f}, 5th pct {np.percentile(null,5):+.4f}, min {null.min():+.4f}; p = share of placebos at least as good as real = {pv:.3f}  => P3 {'PASS' if real < np.percentile(null, 5) else 'FAIL'}")

print("\n=== SECONDARY ===")
print("S1 thresholds (paired vs M0, pooled over folds):")
for nm, f in (("0.5 m/s", "F05"), ("2.0 m/s", "F2")):
    o = run("R0_primary", [f]); paired(o, M0, "model_hit", f"  top-1  thr {nm}"); paired(o, M0, "model_ll", f"  logloss thr {nm}")
print("S2 lagged velocity control:")
ML = run("R0_primary", ["FL"]); paired(ML, M0, "model_hit", "  top-1  lagged - M0"); paired(ML, M0, "model_ll", "  logloss lagged - M0")
paired(M1, ML, "model_hit", "  top-1  current - lagged"); paired(M1, ML, "model_ll", "  logloss current - lagged")
print("S3 other candidate rules (paired M1-M0):")
for rule in ("R_loose", "R_existing", "R_tight"):
    a0, a1 = run(rule, []), run(rule, ["F1"]); paired(a1, a0, "model_hit", f"  {rule} top-1"); paired(a1, a0, "model_ll", f"  {rule} logloss")
print("S5 moving passers only (top-1 M0 -> M1, n):")
for g in (1, 2, 3): mv = M0[g]["moving"]; print(f"  G{g}: n={mv.sum()}  {M0[g]['model_hit'][mv].mean():.3f} -> {M1[g]['model_hit'][mv].mean():.3f}   (static passers n={(~mv).sum()}: {M0[g]['model_hit'][~mv].mean():.3f} -> {M1[g]['model_hit'][~mv].mean():.3f}; F1 is zero there)")
sub = Cd[Cd.R0_primary]; print(f"S6 passers moving >=1 m/s: {(sp_cur >= 1.0).mean()*100:.1f}% of passes; candidate rows with F1 != 0: {(sub.F1 != 0).mean()*100:.1f}%")
