"""DQ-v3: option 2 (in-view training with censoring at leaving view) then option 1 (Metrica LOGO recalibration). PREREG_DQ_v3.txt."""
import sys, os; D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, pandas as pd
from tfcommon import FEATS
import tf_eval as E      # reuses fit, ll, slope, report exactly (module runs v2 on import: prints v2 results again)
S = E.S.sort_values(["pid", "s"], kind="stable").reset_index(drop=True); M = E.M
print("\n" + "=" * 30 + " OPTION 2 " + "=" * 30)
iv = S.in_view.to_numpy() > 0.5; first_out = pd.Series(np.where(~iv, S.s, np.inf)).groupby(S.pid.to_numpy()).transform("min").to_numpy()
keep = S.s.to_numpy() < first_out; S2 = S[keep].copy()
lost_event = S[(~keep) & (S.event == 1)].pid.nunique(); censored = S.groupby("pid").apply(lambda d: (d.in_view < 0.5).any()).sum()
print(f"in-view rows kept {keep.mean()*100:.1f}% ({keep.sum()} of {len(S)}); passes that leave the view (censored there) {censored} of {S.pid.nunique()}; failures lost to censoring {lost_event} of {int(S.event.sum())}")
F2 = [f for f in FEATS if f != "in_view"]
X, y, comp, match = S2[F2].to_numpy(float), S2.event.to_numpy(), S2.comp.to_numpy(), S2.match.to_numpy(); oks = []
for tr, te in (("EU20", "WC22"), ("WC22", "EU20")):
    a, b_ = comp == tr, comp == te; f = E.fit(X[a], y[a]); print(f"train {tr} -> test {te} (in-view rows)"); oks.append(E.report(te, y[b_], f(X[b_]), y[a].mean(), match[b_]))
f2 = E.fit(X, y); Xm = M[F2].to_numpy(float); ym = M.event.to_numpy(); gm = M.game.to_numpy(); p2 = f2(Xm)
print("coefficients (option 2): " + ", ".join(f"{n} {c:+.2f}" for n, c in zip(["const"] + F2, f2.b)))
print("Transfer to Metrica (option 2):"); okm = E.report("Metrica pooled", ym, p2, y.mean(), gm, gate_tau=False)
sl2 = E.slope(ym, p2); print(f"   unrounded pooled slope {sl2:.4f}; O/E {ym.sum()/p2.sum():.4f}; per game: " + ", ".join(f"G{g} slope {E.slope(ym[gm==g], p2[gm==g]):.3f} O/E {ym[gm==g].sum()/p2[gm==g].sum():.3f}" for g in (1, 2, 3)))
print(f"OPTION 2 gate: cross-competition {'PASS' if all(oks) else 'FAIL'}; Metrica {'PASS' if okm else 'FAIL'}; prediction 'slope moves toward 1 vs 0.798': {'YES' if abs(sl2-1) < abs(0.798-1) else 'NO'}")
print("\n" + "=" * 30 + " OPTION 1: LOGO recalibration on Metrica " + "=" * 30)
fv2 = E.fit(E.X, E.y); pv2 = fv2(M[FEATS].to_numpy(float))
def recal(p, label):
    lp = np.log(p / (1 - p)); out = np.empty_like(p)
    for g in (1, 2, 3):
        tr = gm != g; Z = np.column_stack([np.ones(tr.sum()), lp[tr]]); b = np.array([0.0, 1.0])
        for _ in range(100):
            q = 1 / (1 + np.exp(-Z @ b)); s = np.linalg.solve((Z * (q * (1 - q))[:, None]).T @ Z, Z.T @ (ym[tr] - q)); b += s
            if np.abs(s).max() < 1e-10: break
        out[gm == g] = 1 / (1 + np.exp(-(b[0] + b[1] * lp[gm == g]))); print(f"   [{label}] fold G{g}: fitted a={b[0]:+.3f} b={b[1]:.3f}")
    sl, oe = E.slope(ym, out), ym.sum() / out.sum(); ok = 0.8 <= sl <= 1.2 and 0.85 <= oe <= 1.15
    print(f"   [{label}] pooled out-of-fold slope {sl:.4f}, O/E {oe:.4f}; per game " + ", ".join(f"G{g} slope {E.slope(ym[gm==g], out[gm==g]):.3f} O/E {ym[gm==g].sum()/out[gm==g].sum():.3f}" for g in (1, 2, 3))
          + f"; log loss per row before {E.ll(ym, p).mean():.5f} after {E.ll(ym, out).mean():.5f}  => {'PASS' if ok else 'FAIL'}")
    q = pd.qcut(out, 5, labels=False, duplicates="drop"); print("   reliability: " + ", ".join(f"{out[q==k].mean():.4f}->{ym[q==k].mean():.4f}" for k in np.unique(q))); return ok
ok1 = recal(p2, "option-2 model"); recal(pv2, "v2 model (comparison)")
primary = okm if okm else ok1
print(f"\nGATE G2-cal (v3): {'PASS' if primary else 'FAIL'}  (option 2 Metrica {'PASS' if okm else 'FAIL'}; option 1 on option-2 model {'PASS' if ok1 else 'FAIL'})")
pd.to_pickle(dict(b=f2.b, mu=f2.mu, sd=f2.sd, feats=F2), f"{D}/tf_model_v3.pkl")
