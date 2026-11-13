import numpy as np
STEP = 2.0
FEATS = ["s", "s2", "dmin", "ldmin", "ratio", "n3", "mx", "my", "ppress", "in_view"]
def inpoly(pts, poly):
    """ray casting; poly flat [x1,y1,x2,y2,...]"""
    if poly is None or len(poly) < 6: return np.ones(len(pts), bool)
    P = np.asarray(poly, float).reshape(-1, 2); x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = P[:, 0][None], P[:, 1][None]; x2, y2 = np.roll(P[:, 0], -1)[None], np.roll(P[:, 1], -1)[None]
    c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1); return (c.sum(1) % 2) == 1
def steps(a, end, opps, failed, poly=None):
    """rows (features + event) for one pass; a, end: origin/stop point; opps (n,2) at release."""
    v = end - a; L = float(np.hypot(*v)); u = v / L; n = int(np.ceil(L / STEP)); s = np.arange(n) * STEP
    P0 = a + s[:, None] * u; P1 = a + (s[:, None] + STEP) * u; mid = (P0 + P1) / 2
    if len(opps):
        ab = P1 - P0; t = np.clip(np.einsum("kj,okj->ok", ab, opps[:, None, :] - P0[None]) / (ab * ab).sum(1)[None], 0, 1)
        dd = np.hypot(*(P0[None] + t[..., None] * ab[None] - opps[:, None, :]).transpose(2, 0, 1))      # (opp, step)
        dmin = np.minimum(dd.min(0), 20.0); n3 = (dd <= 3).sum(0); pp = min(float(np.hypot(*(opps - a).T).min()), 20.0)
    else:
        dmin = np.full(n, 20.0); n3 = np.zeros(n); pp = 20.0
    ev = np.zeros(n); ev[-1] = float(failed)
    return np.column_stack([s, s ** 2, dmin, np.log1p(dmin), dmin / (s + STEP), n3, mid[:, 0], np.abs(mid[:, 1] - 40), np.full(n, pp), inpoly(mid, poly).astype(float), ev])
