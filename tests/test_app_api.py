"""The local app's scoring API (app/api.py)."""
import os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from api import score_situation

BASE = dict(carrier=[100, 40], teammates=[[108, 38], [104, 52]], opponents=[[110, 42], [106, 36], [118, 40]], keeper=[118, 40])

def test_options_are_ranked_thresholded_and_capped():
    r = score_situation(dict(BASE, threshold=0, top=10)); q = [o["q"] for o in r["options"]]
    assert q == sorted(q, reverse=True) and len(r["options"]) <= 10 and r["zone"] == "attacking"
    assert {o["kind"] for o in r["options"]} <= {"shot", "pass"} and all(0 <= o["score"] <= 100 for o in r["options"])
    hi = score_situation(dict(BASE, threshold=90)); assert hi["options"] == [] and hi["n_options"] >= 1

def test_shot_only_from_x60_and_zone_warning():
    r = score_situation(dict(carrier=[40, 40], teammates=[[50, 40]], opponents=[[60, 40], [45, 30]], threshold=0))
    assert all(o["kind"] != "shot" for o in r["options"]) and r["zone"] == "middle" and any("not validated" in w for w in r["warnings"])

def test_certain_offside_hidden_and_bad_input_rejected():
    r = score_situation(dict(carrier=[90, 40], teammates=[[118, 20]], opponents=[[100, 40], [101, 30], [102, 50]] + [[95, 5 + 7 * i] for i in range(7)], keeper=[119, 40], threshold=0))      # 11 opponents marked: a pass to x = 118 is certainly offside
    assert r["n_offside"] == 1 and all(o["kind"] != "pass" for o in r["options"])
    for bad in (dict(), dict(carrier=[100, 40]), dict(carrier=[130, 40], opponents=[[1, 1]])):
        with pytest.raises(ValueError): score_situation(bad)
