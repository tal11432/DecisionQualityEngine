# IDSSE one-match pilot (J03WPY): stopped at the critical release-frame gate

**Result: the pilot did not complete. Phase 2's critical gate failed, so Phase 3 (label comparison) and Phase 4 (feature generation) were not run, as instructed.**
**Recommendation for all seven matches: NO-GO under the frozen methodology.**

Reproduce (after the three files are in `data/idsse/J03WPY/`): `python -m src.idsse_pilot --out reports/idsse_pilot_results.txt` (exit code 1 = gate failed; ~20 s).
Raw output: `reports/idsse_pilot_results.txt`. Nothing is committed: per the task, commit and push only happens on a successful pilot.

## 1. Phase 1: download and ingestion

**Access.** The sandbox's network policy blocked Hugging Face. Two allowlist changes were proposed and approved on cards: `huggingface.co`, then `us.aws.cdn.hf.co` (where the tracking file is served). They live in a new personal environment `decisionqualityengine` that this session was moved onto; they persist for future sessions on that environment and can be removed in its settings. Moving this session to that environment also means the old environment's Figma API credential is no longer sent from this session.

Exactly the three J03WPY files were downloaded into the git-ignored `data/idsse/J03WPY/` (`MANIFEST.json` there records the same):

| File | Bytes | SHA-256 |
|---|---|---|
| `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml` | 11,922 | `566dd5e4c86e124ab90fcdba51100881f0436e22afa71832945a451a2d95a420` |
| `DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml` | 698,898 | `e440a3adf9a38d3e4ce4289343a59a3b5f76018a78e97c765114f7817591fb3c` |
| `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml` | 371,688,499 | `845202c29ad93d696653436da426a7cdaefb848298e07b7343de6529581671e1` |

URL pattern: `https://huggingface.co/datasets/pysport/idsse-data/resolve/main/<file name>`. The tracking file is 372 MB (my earlier estimate of 430-570 MB was too high). The license is still **unverified from an official page** (nature.com/figshare unreachable); the only statement seen is CC-BY 4.0 in kloppy's loader docstring.

**Adapter** (`src/ingestion/dfl.py`, ~200 lines, standard library + numpy/pandas): missing X/Y are NaN (kloppy would default to 0); the frame index is the union over all objects, so no frame or person is dropped; `time_s` is built from each frame's real `T` and each event's real `EventTime` on one clock (no fixed frame offsets); raw event attributes are kept (flattened JSON in `raw`, element chain in `chain`); pitch size comes from the match information and is cross-checked against the positions file. Parsing takes ~15 s. Tests: `tests/test_idsse.py` (10, synthetic match): NaN preservation, no dropped frames/players, timestamps vs frame offsets, orientation, pitch mismatch, raw attributes, frozen eligibility, label definitions, gate logic. Full suite: **127 passed**. No existing file was modified.

## 2. Phase 2: data quality (all verified on the real files)

| Check | Result |
|---|---|
| Frequency / continuity | exactly **25.00 Hz**; 0 irregular steps, 0 frame-number gaps, 0 duplicate frames; frame `T` agrees across all 55 object streams (0 disagreements) |
| Periods | P1 frames 10,000-79,130 (0-2,765.2 s, 69,131 frames); P2 frames 100,000-177,079 (3,720.6-6,803.8 s, 77,080 frames); durations equal the metadata's TotalTimeFirstHalf/SecondHalf (2,765.2 s / 3,083.16 s) |
| Missing data | **0 NaN coordinates, 0 frames without a ball record**; the ball position is present even while dead (`BallStatus` alive 81,460 / dead 64,751 frames). The "29% player missing" in the existing report is the 9 off-pitch persons among 31 columns: exactly 22 players are tracked in every frame |
| Pitch / coordinates | 105 x 68 (match information = positions file); metric, centre origin, x -55.5..55.5, y -36.9..37.9 (250 values beyond pitch+3 m); orientation: inferred attack direction agrees with the provider's `KickOff TeamLeft` (Nürnberg left in half 1) |
| Identities | 31 persons tracked; 9 of 40 listed players never played; none tracked but unlisted; of 891 `Play` events: passer not tracked 0, passer on the wrong team 0, recipient not tracked 0, recipient missing 13; identity-swap detector (built for Metrica Game 3): none |
| Substitutions / lineup | 9 substitutions; in every swap the outgoing player's last frame is the incoming player's first; the substitution *event* precedes the tracked swap by 0-22 s |
| Event sync (0.02 s) | 889/890 `Play` events matched; 1 outside both periods |
| Existing validation report | PASS (warnings: coordinates beyond margin 250, speed jumps 189 player / 90 ball) |

The tracking itself is in good shape and cleaner than Metrica's (no ball gaps). The problem is the **event timing**.

## 3. Critical gate: does `EventTime` identify the pass release?

**Population:** the frozen-eligible passes: open-play `Play/Pass` only. Of 891 `Play` events, 760 are eligible; excluded: ThrowIn 41, cross 20, FreeKick 20, through ball 18, GoalKick 18, CornerKick 11, KickOff 3. **"Exclude headers" cannot be applied**: IDSSE passes carry no header flag (`head`/`header` occur only on shot and assist attributes), and no proxy rule was invented, so header passes remain in the eligible set.

**Method.** For each pass: ball-to-passer distance at the frame nearest `EventTime` (d0); release lag = time at which the ball first leaves 1 m of the passer, searched from -0.48 s; mean ball speed over the next 6 frames. The identical code was run on Metrica Game 1, where the existing decision-frame definition (release = event start frame) was validated. The five pass criteria were fixed in code **before** measuring.

| Measure | IDSSE J03WPY (n=760) | Metrica G1 (n=883) | Criterion | |
|---|---|---|---|---|
| median d0 | **2.40 m** | 0.09 m | <= 0.5 m | FAIL |
| share with ball > 3 m from the passer | **47.8%** | 1.0% | <= 3% | FAIL |
| median release lag [IQR] | -0.24 s [-0.35, -0.04] (74% censored: ball already > 1 m away) | +0.12 s [0.12, 0.16] | within 0.08 s; IQR <= 2x | FAIL |
| median ball speed after the event | **5.5 m/s** | 11.5 m/s | >= 8 m/s | FAIL |

Lag profile (median over passes): Metrica shows a sharp release signature between lag 0 and +0.08 s (distance 0.09 -> 0.65 m, speed 2.9 -> 12.2 m/s). **IDSSE is flat across +-0.4 s** (distance 3.6 -> 2.4 -> 2.4 m, speed 5.9 -> 5.5 -> 5.4 m/s): no release signature at any lag.

**Diagnosis (no re-timing performed).** This is not a constant clock offset. The per-event lag at which the ball is closest to the named passer (within +-5 s) has median -0.38 s and IQR -1.72 to +0.64 s, and only **7%** of passes fall within +-3 frames. The wide profile is symmetric around lag 0 (median distance 2.4 m at 0 s; 5.2 m at +-1 s; 15.4 m at -4 s and 17.7 m at +4 s), so `EventTime` is near the contact on average but imprecise by seconds, and the error varies from pass to pass. Other evidence in the file points the same way: events carrying provider frame references differ from `EventTime` by up to 6 s (`StartFrame` n=15, median +2.7 s, on set pieces; `CalculatedFrame` n=25, median +0.3 s, range -2.1..+5.9 s, on shots), and substitution events precede the tracked swap by up to 22 s.

**Consequence.** The existing decision-time definition requires the frame at which the ball leaves the passer's feet (validated on Metrica to within a frame). `EventTime` cannot supply that for IDSSE, so scenarios A and B cannot be built on it without **redefining the decision frame**, which the frozen methodology forbids doing silently. **Stopped here.**

## 4. Not run, and why

* **Phase 3 (label agreement):** not run; it follows a failed critical gate. `src/idsse_labels.py` implements both definitions and the frozen eligibility rules and is unit-tested on a synthetic match, but has **not** been executed on the real match: label agreement, counts, interception rates and examples are therefore unknown. One descriptive fact from the raw file (not a Phase 3 result): `BallClaiming Type=InterceptedBall` occurs only 4 times in the whole match, against 166 unsuccessful `Play` events, so definition B could label at most 4 interceptions; a material disagreement with definition A would be near certain if the pilot had continued.
* **Phase 4 (feature pipeline, A/B compatibility, eligibility and exclusion counts for the features):** not run. What can be said without running it: IDSSE is 25 Hz and metric, so the frame-based windows (5 frames, 6 frames) and distances would keep their meaning; the only blocked element is the decision frame.

## 5. Incompatibilities with the frozen methodology

1. **Decision frame (blocking):** `EventTime` is not release-accurate (section 3).
2. **Headers:** cannot be identified for passes; the frozen exclusion is not implementable.
3. **Interception label:** no provider interception flag for passes; both candidate definitions are derived and untested on data (B is very sparse in this match).
4. **"Positive duration" label sanity check (Metrica):** IDSSE events have a single `EventTime`; not applicable.

## 6. Recommendation

**NO-GO for all seven matches** under the frozen methodology, because the property that the decision-time definition depends on is absent from the event file and is a property of the dataset's annotation, not of this one match. Only J03WPY was approved and tested; the other six cannot be assumed to behave identically, but nothing suggests they would.

Options requiring explicit approval (none attempted):
1. **Tracking-derived release detection** for IDSSE (e.g. the frame where the ball leaves the passer), pre-registered and validated against Metrica; this *redefines how the decision frame is obtained* and changes eligibility (only passes with a detectable release), so it is a new methodology, not a pilot continuation.
2. Move to a different source (the audit's candidates) whose events carry release-accurate timing, applying the same gate first.
3. Use IDSSE only for descriptive cross-checks that do not depend on the decision frame.
