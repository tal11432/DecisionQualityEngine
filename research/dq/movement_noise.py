"""How far do players move in 0.4 s and 1 s? (Metrica tracking, 25 Hz, 3 games, 4,000 random frames per game.) Used to choose the position noise of the floor calibration from data instead of assuming it."""
import sys; sys.path.insert(0, "."); import numpy as np
from src.config import load_config
from src.cli import _load
cfg = load_config(); rng = np.random.default_rng(0)
for dt_s in (0.4, 1.0):
    allv, near = [], []
    for g in (1, 2, 3):
        td, ev = _load(cfg, g); f = td.frames; k = int(round(dt_s * 25)); cols = list(td.players[["team", "player_id"]].itertuples(index=False, name=None))
        X = np.column_stack([f[td.col(t, p, "x")].to_numpy(float) for t, p in cols]); Y = np.column_stack([f[td.col(t, p, "y")].to_numpy(float) for t, p in cols]); bx, by = f["ball_x"].to_numpy(float), f["ball_y"].to_numpy(float)
        idx = rng.choice(np.arange(5000, len(f) - k - 1), 4000, replace=False); d = np.hypot(X[idx + k] - X[idx], Y[idx + k] - Y[idx]); db = np.hypot(X[idx] - bx[idx][:, None], Y[idx] - by[idx][:, None]); ok = np.isfinite(d)
        allv.append(d[ok]); near.append(d[ok & (db <= 25)])
    a, n = np.concatenate(allv), np.concatenate(near)
    print(f"window {dt_s}s: any player median {np.median(a):.2f} m, mean {a.mean():.2f}, 90th {np.percentile(a,90):.2f}; players within 25 m of the ball: median {np.median(n):.2f}, mean {n.mean():.2f}, 90th {np.percentile(n,90):.2f}; implied per-axis sd {n.mean()/1.2533:.2f} m")
