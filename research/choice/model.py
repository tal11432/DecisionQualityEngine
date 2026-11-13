"""Stage 2: conditional-logit choice model + baselines, leave-one-game-out (rules frozen in PREREGISTRATION.txt)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.pass_completion import Prep
FEATS = ["dist", "forward", "lateral", "rec_press", "margin"]
LAMS = (0.1, 1.0, 10.0, 100.0)
Cd, Pd, RULES = pd.read_pickle(sys.argv[1]); Pd = pd.read_pickle(sys.argv[2])
Pd = Pd.set_index("pass")

def sets_for(rule, games, conditional=True):
    d = Cd[Cd[rule] & Cd.game.isin(games)].sort_values(["pass"], kind="stable")
    g = d.groupby("pass")
    ok = g.is_actual.any() & (g.size() >= 2) if conditional else (g.size() >= 1)
    d = d[d["pass"].isin(ok[ok].index)].reset_index(drop=True)
    starts = np.flatnonzero(np.r_[True, d["pass"].to_numpy()[1:] != d["pass"].to_numpy()[:-1]])
    return d, starts

def fit_clogit(X, chosen, starts, lam, iters=50):
    n, p = X.shape; gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
    b = np.zeros(p)
    for _ in range(iters):
        u = X @ b; mx = np.maximum.reduceat(u, starts); e = np.exp(u - mx[gid]); s = np.add.reduceat(e, starts); pr = e / s[gid]
        m = np.add.reduceat(pr[:, None] * X, starts, axis=0)
        grad = X[chosen].sum(0) - m.sum(0) - lam * b
        XX = pr[:, None, None] * X[:, :, None] * X[:, None, :]
        H = -(np.add.reduceat(XX, starts, axis=0).sum(0) - np.einsum("gi,gj->ij", m, m)) - lam * np.eye(p)
        step = np.linalg.solve(H - 1e-9 * np.eye(p), grad); b = b - step
        if np.abs(step).max() < 1e-8: break
    return b

def probs(X, b, starts):
    n = len(X); gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n])); u = X @ b
    e = np.exp(u - np.maximum.reduceat(u, starts)[gid]); return e / np.add.reduceat(e, starts)[gid][...]

def chosen_idx(d, starts):
    return np.flatnonzero(d.is_actual.to_numpy())  # exactly one per set (conditional=True)

def metrics_from_scores(score, d, starts, higher_better=True):
    """Per-pass (hit, reciprocal rank) from candidate scores; ties handled by averaging."""
    n = len(d); ends = np.r_[starts[1:], n]; ch = chosen_idx(d, starts); hit, rr = [], []
    for s, e, c in zip(starts, ends, ch):
        sc = score[s:e] * (1 if higher_better else -1); me = sc[c - s]; others = np.delete(sc, c - s)
        gt, eq = (others > me).sum(), (others == me).sum()
        hit.append(1.0 / (1 + eq) if gt == 0 else 0.0); rr.append(1.0 / (1 + gt + 0.5 * eq))
    return np.array(hit), np.array(rr)

def evaluate(rule, tr_games, te_game):
    dtr, str_ = sets_for(rule, tr_games); dte, ste = sets_for(rule, [te_game])
    prep = Prep().fit(dtr[FEATS].to_numpy(float)); Xtr = prep.transform(dtr[FEATS].to_numpy(float)); Xte = prep.transform(dte[FEATS].to_numpy(float))
    # inner leave-one-training-game-out selection of the ridge penalty (training games only)
    best, bl = None, np.inf
    for lam in LAMS:
        ll = []
        for hold in tr_games:
            a, sa = sets_for(rule, [g for g in tr_games if g != hold]); h, sh = sets_for(rule, [hold])
            pr_ = Prep().fit(a[FEATS].to_numpy(float)); bb = fit_clogit(pr_.transform(a[FEATS].to_numpy(float)), chosen_idx(a, sa), sa, lam)
            p = probs(pr_.transform(h[FEATS].to_numpy(float)), bb, sh); ll.append(-np.log(p[chosen_idx(h, sh)]).mean())
        if np.mean(ll) < bl: best, bl = lam, np.mean(ll)
    b = fit_clogit(Xtr, chosen_idx(dtr, str_), str_, best)
    p = probs(Xte, b, ste); ch = chosen_idx(dte, ste); n_sizes = np.diff(np.r_[ste, len(dte)])
    out = {"n": len(ste), "sizes": n_sizes, "lam": best, "beta": b, "prep": prep, "pass": dte["pass"].to_numpy()[ste]}
    out["model_ll"] = -np.log(p[ch]); out["model_hit"], out["model_rr"] = metrics_from_scores(p, dte, ste)
    out["near_hit"], out["near_rr"] = metrics_from_scores(dte.dist.to_numpy(), dte, ste, higher_better=False)
    out["safe_hit"], out["safe_rr"] = metrics_from_scores(dte.margin.to_numpy(), dte, ste, higher_better=False)
    H = np.array([np.sum(1.0 / np.arange(1, k + 1)) for k in n_sizes])
    out["unif_hit"], out["unif_rr"], out["unif_ll"] = 1.0 / n_sizes, H / n_sizes, np.log(n_sizes)
    # velocity-aligned (existing rule) on moving passers
    mv = (Pd.loc[out["pass"], "speed"].to_numpy() >= 1.0); out["moving"] = mv
    out["vel_hit"], out["vel_rr"] = metrics_from_scores(dte.cos_vel.to_numpy(), dte, ste)
    out["maxp"] = np.maximum.reduceat(p, ste)
    return out

def boot(arrs, idx):  # arrs: dict name -> per-pass array
    return {k: np.array([v[i].mean() for i in idx]) for k, v in arrs.items()}

def ci(x): return f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    allres = {}
    for rule in RULES:
        print(f"\n################ {rule} {RULES[rule]} ################")
        for te in (1, 2, 3):
            tr = [g for g in (1, 2, 3) if g != te]; r = evaluate(rule, tr, te); allres[(rule, te)] = r
            n = r["n"]; idx = rng.integers(0, n, (1000, n)); mv = r["moving"]
            B = boot({"m_hit": r["model_hit"], "m_rr": r["model_rr"], "m_ll": r["model_ll"], "n_hit": r["near_hit"], "n_rr": r["near_rr"], "s_hit": r["safe_hit"], "s_rr": r["safe_rr"],
                      "u_hit": r["unif_hit"], "u_rr": r["unif_rr"], "u_ll": r["unif_ll"]}, idx)
            print(f"\nheld-out G{te}: train G{tr[0]}+G{tr[1]}; n passes {n} (mean set size {r['sizes'].mean():.2f}); ridge {r['lam']}")
            print(f"  {'method':<28}{'top-1':>26}{'MRR':>26}{'log loss':>26}")
            f = lambda nm, h, rr_, ll=None: print(f"  {nm:<28}{h.mean():>10.3f} {ci(B[nm[0] + '_hit'] if False else np.array([h[i].mean() for i in idx])):>15}{rr_.mean():>10.3f} {ci(np.array([rr_[i].mean() for i in idx])):>15}" + (f"{ll.mean():>10.3f} {ci(np.array([ll[i].mean() for i in idx])):>15}" if ll is not None else f"{'n/a':>26}"))
            f("uniform (analytic)", r["unif_hit"], r["unif_rr"], r["unif_ll"])
            f("nearest teammate", r["near_hit"], r["near_rr"])
            f("safest lane (min margin)", r["safe_hit"], r["safe_rr"])
            f("conditional logit (5 feat.)", r["model_hit"], r["model_rr"], r["model_ll"])
            print(f"  velocity-aligned (moving passers, n={mv.sum()}): top-1 {r['vel_hit'][mv].mean():.3f}, MRR {r['vel_rr'][mv].mean():.3f}; model on same subset {r['model_hit'][mv].mean():.3f} / {r['model_rr'][mv].mean():.3f}; uniform {r['unif_hit'][mv].mean():.3f} / {r['unif_rr'][mv].mean():.3f}")
            d = lambda a, b_: a - b_
            print(f"  paired model - uniform: top-1 {d(B['m_hit'], B['u_hit']).mean():+.3f} {ci(d(B['m_hit'], B['u_hit']))}, MRR {d(B['m_rr'], B['u_rr']).mean():+.3f} {ci(d(B['m_rr'], B['u_rr']))}, log loss {d(B['m_ll'], B['u_ll']).mean():+.3f} {ci(d(B['m_ll'], B['u_ll']))}")
            print(f"  paired model - nearest: top-1 {d(B['m_hit'], B['n_hit']).mean():+.3f} {ci(d(B['m_hit'], B['n_hit']))}, MRR {d(B['m_rr'], B['n_rr']).mean():+.3f} {ci(d(B['m_rr'], B['n_rr']))}")
    pd.to_pickle(allres, sys.argv[3])
