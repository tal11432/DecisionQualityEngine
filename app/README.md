# Decision Score app (local)

```bash
python app/server.py        # then open http://127.0.0.1:8000
```

Upload a screenshot (optional, it only stays in your browser), choose which part of the pitch it shows, and mark the ball carrier, teammates, opponents and the opposing keeper by clicking (drag to move, right-click to delete). The team in possession attacks to the right. The app scores every option it can see (a shot from x >= 60 and ground passes of 3-40 m to the marked teammates) with the research model and lists up to 10 above the score threshold.
Read the limits in `reports/decision_score_v1.md`: the score is validated in the attacking third only; carries, lofted/through balls, crosses and dribbles are not options; no expert validation.
