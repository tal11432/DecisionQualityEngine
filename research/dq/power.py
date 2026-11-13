"""Expert-panel design: precision of Krippendorff's alpha (interval) and power for DQ-expert correlation. Simulation, seed 0."""
import numpy as np
rng = np.random.default_rng(0)
def kalpha(R):  # complete data, interval metric: R (units, raters)
    n, m = R.shape; Do = np.mean([((R[:, i] - R[:, j]) ** 2).mean() for i in range(m) for j in range(m) if i < j]) 
    v = R.ravel(); De = ((v[:, None] - v[None, :]) ** 2).sum() / (len(v) * (len(v) - 1)); return 1 - Do / De
def sim(n, m, rel):  # latent quality ~N(0,1); rater noise so that single-rater reliability = rel; 5-point ordinal cut
    q = rng.standard_normal(n); sd = np.sqrt(1 / rel - 1); X = q[:, None] + sd * rng.standard_normal((n, m))
    return np.digitize(X, np.quantile(X, [0.2, 0.4, 0.6, 0.8])) + 1, q
print("Krippendorff alpha (interval), 5-point ratings; 2,000 simulated panels per cell: true single-rater reliability -> estimate mean [2.5%, 97.5%]")
for rel in (0.3, 0.5):
    for m in (3, 5):
        for n in (60, 120, 200):
            a = [kalpha(sim(n, m, rel)[0]) for _ in range(2000)]
            print(f"  rel {rel} raters {m} situations {n}: alpha {np.mean(a):.2f} [{np.percentile(a,2.5):.2f}, {np.percentile(a,97.5):.2f}]")
from math import atanh, ceil
for rho in (0.2, 0.3, 0.4):
    print(f"n to detect correlation {rho} (two-sided 5%, 80% power, Fisher z): {ceil(((1.96 + 0.8416) / atanh(rho)) ** 2 + 3)}")
# attenuation: observed corr between DQ and the MEAN expert rating if DQ correlates r_true with latent quality
for rel, m in ((0.3, 3), (0.5, 3), (0.5, 5)):
    relmean = m * rel / (1 + (m - 1) * rel); print(f"reliability of the mean of {m} raters at single-rater {rel}: {relmean:.2f} -> observed corr = true corr x {np.sqrt(relmean):.2f}")
