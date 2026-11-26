# Decision Score app

**Static site (no server, no install):** open `app/static/index.html` in a browser, or publish the folder `app/static/` as is (for example GitHub Pages: Settings -> Pages -> deploy from a branch -> folder). Everything is computed in the browser by `engine.js`; the screenshot and the marks never leave the page.

Upload a screenshot (optional), choose which part of the pitch it shows or calibrate the perspective from known pitch corners (4-5 clicks), and mark the ball carrier, teammates, opponents and the opposing keeper (drag to move, right-click to delete). The team in possession attacks to the right. The app scores every option it can see (a shot from x >= 60 and ground passes of 3-40 m to the marked teammates) and lists up to 10 above the score threshold; click rows in the table to draw them on the image.

## Files
* `static/index.html`, `static/engine.js` (a port of `research/dq/dscore`), `static/models.js` (generated coefficients, 5 KB).
* `export_models.py` regenerates `models.js` from `research/dq/*.pkl` after any model change.
* `parity.js` + `tests/test_js_parity.py`: the JavaScript engine reproduces the Python scoring on random situations (max difference in Q about 1e-16).
* `server.py`, `api.py`: the optional local Python server (the page does not need it).
* `examples/`: calibration examples on a broadcast screenshot: **do not publish** (the image belongs to the broadcaster).

Read the limits in `reports/decision_score_v1.md`: the score is validated in the attacking third only; carries, lofted/through balls, crosses and dribbles are not options; no expert validation. Models trained on StatsBomb open data (attribution required).
