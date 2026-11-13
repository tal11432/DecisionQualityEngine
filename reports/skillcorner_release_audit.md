# SkillCorner release-frame feasibility audit (one match, one 12 MB tracking slice)

**Recommendation: NO-GO for a one-match tracking pilot.** SkillCorner's event frames and tracking **can** locate the release roughly (the event's `frame_end` is the release, confirmed independently), but not with the precision, coverage or outcome-balance that the existing decision-time pipeline needs. Four of six pre-registered criteria fail, and no detection threshold satisfies them together.

## 0. Scope and data actually used

* **Events:** `2017461_dynamic_events.csv` (3.96 MB), the match already inspected (A-League, 2025-05-17, Melbourne Victory v Auckland FC).
* **Tracking:** the 89.5 MB LFS object is **not** available locally and the 1.8 GB collection was not touched. I read a **12 MB byte-range prefix** of this one match through `media.githubusercontent.com` (frames 0-9,330 at 10 Hz; play starts at frame 2510), which holds **95 passes** with a full +-2 s window (72 successful, 23 unsuccessful, 29 zero-duration events). Reproduce:
  ```bash
  curl -r 0-11999999 -o trk_head.jsonl https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/2017461/2017461_tracking_extrapolated.jsonl
  curl -O https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/2017461/2017461_dynamic_events.csv
  python -m src.skillcorner_release_audit --events 2017461_dynamic_events.csv --tracking trk_head.jsonl --out reports/skillcorner_release_results.txt
  ```
  (exit code 1 = not all criteria met). Raw output: `reports/skillcorner_release_results.txt`. No model, no features, no existing pipeline file touched; `src/skillcorner_release_audit.py` is new and `tests/test_skillcorner_release.py` (6 tests, synthetic) covers the detection rule. Full suite: 133 passed.
* **Verified tracking structure:** one JSON line per frame, consecutive integer frames, `timestamp` match clock, `ball_data{x,y,z,is_detected}`, `player_data[{x,y,player_id,is_detected}]` (22 players per frame), possession flag. In this slice the ball is **detected in 64.6%** of frames and player positions in **42.9%**; the rest are extrapolated.

## 1. Method and criteria (fixed before the first measurement)

Release is detected from **tracking only**: `inside(t) = |ball - passer| <= 1.5 m`; a departure at `k` needs `inside(k-1)` and not inside at `k, k+1, k+2` (look-ahead 2 frames = 0.2 s, within the 0.24 s Scenario-B window); release `r = k-1` (last frame in contact). It counts only if the passer **and** ball are *detected* (not extrapolated) in `k-1..k+2`. The event frames are used only to locate a +-2 s search window around the pass; passes with several candidate departures are reported as **ambiguous and never resolved with event frames**, so tracking is not merely confirming event labels. Reference: the Metrica release signature (ball 0.09 m from the passer at the event frame, only 1% of events with the ball > 3 m away, release within about 0.08 s of the event frame at 25 Hz).

| Criterion | Threshold | Result | |
|---|---|---|---|
| C1 identifiable release | >= 80% of passes | **65.3%** (62/95; 95% Wilson 55-74%) | FAIL |
| C2 release within 1 frame (0.1 s) of an event frame | >= 80% | `frame_end`: **74.2%** | FAIL |
| C3 completed vs unsuccessful identifiable share | gap <= 10 pp | 69.4% vs 52.2% = **17 pp** | FAIL |
| C4 zero-duration events | identifiable within 10 pp of others, median error <= 1 frame | 44.8% vs 74.2% = **29 pp** | FAIL |
| C5 look-ahead 1 vs 2 frames | same release in >= 90% | 98.4% (61/62) | PASS |
| C6 pass-like, no deflection | >= 80% | 93.5% | PASS |

## 2. Answers

**1. Do the events carry release-precise frame references?** Partly. Independent tracking shows that for passes the event's **`frame_end` is the release** (tracking-detected release minus `frame_end`: median 0, IQR -1..+1 frames, 34% exactly 0, 74% within +-1 frame) and **`frame_start` is the possession start, i.e. the reception** (median +12 frames = 1.2 s earlier; only 14.5% within +-1 frame). `frame_physical_start` is empty in this match. The event frames are integers at 10 Hz and therefore cannot be better than +-0.05 s even if perfect.

**2. Can release be identified independently from ball-passer distance and ball speed?** For **65%** of passes (62/95). The rest: no departure found 19 (20%), ambiguous (several departures in the window) 12 (13%), passer or ball extrapolated around the departure 2 (2%). The "no departure" cases are **not** an extrapolation problem (their passer and ball are detected in a median 100% of window frames); they come from **positional accuracy**: at `frame_end` the ball is a median **1.39 m** from the passer (Metrica: 0.09 m) and 7.7% of passes (detected positions only) exceed 3 m (Metrica 1%; the IDSSE failure was 47.8%), so a 1.5 m contact radius sits at the measurement noise.

**3. What timing accuracy is achievable at 10 Hz?** Quantisation alone is +-0.05 s (Metrica 25 Hz: +-0.02 s). Measured where identifiable: median 0 frames, IQR +-1 frame, but heavy tails (errors of -15, -12, -12, -9, -8, +7, +14 frames in 7 of 62 passes) and 26% outside +-0.1 s. Part of the +1 frame mass is a rule artefact (the "last frame within 1.5 m" can trail the geometric release by one frame for fast passes). This is at best a ~0.1 s resolution with 74% reliability, versus ~0.04-0.08 s for Metrica.

**4. Zero-duration events.** 29 of 95 passes (31%) have `frame_start == frame_end` (one-touch passes). Only **44.8%** of them have an identifiable release (74.2% for others) and their median error is +1 frame (+-1-frame share 69% vs 76%). With release and reception in the same 0.1 s frame, arrival and departure cannot be separated at 10 Hz (9.7% of identified passes also have <= 1 contact frame). Unsuccessful passes are over-represented among them.

**5. Release vs reception, deflection or another touch.** Among the *identified* departures 93.5% are pass-like (>= 5 m/s over 0.3 s; median 9.8 m/s) and 0% show a sharp deflection (> 60 degrees within 0.5 s), and the median hold before release is 10 frames. But the rule cannot tell a pass from a hard dribble touch except by speed, cannot separate reception from release when the contact phase is <= 1 frame, and was not cross-checked against the recipient's arrival. It says nothing about the 35% of passes it cannot identify.

**6. Future information.** Yes, within the Scenario-B budget: detection needs only 2 frames after the last contact (0.2 s <= 0.24 s), and using 1 frame gives the same release frame in 98.4% of cases. (At 10 Hz the 6-frame/0.24 s window is 2.4 frames, so the frame-based definition itself would need a seconds-based profile.)

## 3. Selection bias (the decisive problem)

Identifiability depends strongly on the outcome and event type: successful 69.4% vs unsuccessful **52.2%**; zero-duration **44.8%** vs 74.2%. The unsuccessful passes (the interceptions that matter) are the ones most often lost: their passer/ball are less often detected (median detected share 0.84 vs 1.00), and they are over-represented among one-touch events. Restricting a pilot to passes with an identifiable release would therefore **bias the completed/intercepted mix and the base rates** (the quantity pooled validation found unstable), in a different direction from Scenario B's existing directedness bias. Samples are small (23 unsuccessful passes): the 17 pp outcome gap is only about 1.4 standard errors, i.e. **not statistically distinguishable** from no gap at this size (the 29 pp zero-duration gap, ~2.5 standard errors, is clearer). Both exceed the pre-registered limit, so the criteria are failed, but the outcome bias is an indication to be confirmed, not an established effect.

## 4. Robustness of the verdict (exploratory, not used for the verdict)

Loosening the contact radius trades coverage against accuracy; no setting meets C1, C2 and C3 together:

| contact radius | identified | within +-1 frame | successful vs unsuccessful | zero-duration vs other |
|---|---|---|---|---|
| 1.5 m (registered) | 65% | 74% | 69% vs 52% | 45% vs 74% |
| 2.0 m | 77% | 77% | 83% vs 57% | 69% vs 80% |
| 2.5 m | 82% | 51% | 86% vs 70% | 76% vs 85% |
| 3.0 m | 85% | 37% | 89% vs 74% | 79% vs 88% |

## 5. Limitations

One match and its first ~16 minutes (95 passes; wide intervals); broadcast tracking with 57% of player positions extrapolated in this slice; later portions of the match could differ; single detection-rule family; no recipient-arrival cross-check; no comparison with the other 19 matches. None of these would plausibly lift detection to Metrica-level precision, because the dominant limits (10 Hz, ~1.4 m positional noise at contact, extrapolation) are properties of the data.

## 6. GO / NO-GO

**NO-GO for a one-match tracking pilot under the existing decision-time definition.** Release timing is only partly independently validated (`frame_end` ~ release to +-1 frame in 74% of identifiable passes), 35% of passes cannot be identified at all, and the missing ones are systematically the unsuccessful and one-touch passes. Per the stop rule, no pilot is justified.

Options needing explicit approval (none attempted): (a) define the decision frame as the event's `frame_end` and report its +-0.1 s uncertainty (a new, weaker definition than the Metrica one; selection-free but unvalidated beyond this slice); (b) use SkillCorner only for descriptive, release-independent analyses; (c) look for a source with release-accurate 25 Hz event timing (IDSSE failed the same type of gate).
