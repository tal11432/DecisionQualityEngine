"""The JavaScript engine of the static app (app/static/engine.js) reproduces the Python scoring (app/api.py -> research/dq/dscore) on random situations."""
import json, os, shutil, subprocess, sys
import numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
from api import score_situation

def _cases(n=250, seed=0):
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        ax = float(rng.uniform(25, 118)); a = [ax, float(rng.uniform(8, 72))]
        mates = [[float(np.clip(ax + rng.uniform(-30, 25), 1, 119)), float(rng.uniform(5, 75))] for _ in range(int(rng.integers(1, 7)))]
        opps = [[float(np.clip(ax + rng.uniform(-25, 22), 1, 119)), float(rng.uniform(5, 75))] for _ in range(int(rng.integers(3, 11)))]
        inp = dict(carrier=a, teammates=mates, opponents=opps, threshold=0, top=10)
        if rng.random() < 0.8: inp["keeper"] = [float(rng.uniform(112, 119.5)), float(rng.uniform(34, 46))]
        out.append(dict(id=i, input=inp, output=score_situation(inp)))
    return out

@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_javascript_engine_matches_python(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "app", "export_models.py")], check=True, capture_output=True)
    f = tmp_path / "cases.json"; f.write_text(json.dumps(_cases()))
    r = subprocess.run(["node", os.path.join(ROOT, "app", "parity.js"), str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
