# Phase 12 — Classroom Analytics

The final demo of the roadmap. Everything from Phases 1–11 is wired into one
application: it watches a classroom (recorded video or a live RTSP camera),
finds the people in it, keeps a stable identity on each one, and writes down
what happened so it can be analysed later.

Detection is **YOLO11-nano**, tracking is **BoT-SORT + ReID**, and on top of
that sits a second-stage **Kalman filter** plus a student registry that decides
who is real, who is temporarily hidden, and who has left.

Only class 0 (`person`) is ever detected. Nothing else in the room is tracked.

```
Camera / video ─► frame capture ─► preprocessing ─► person detection
                                                          │
                              multi-object tracking (BoT-SORT + ReID)
                                                          │
                       quality gates (area · aspect · ROI · duplicates)
                                                          │
                  student registry (Kalman · coasting · re-identification)
                                                          │
                      zones ─ movement ─ statistics ─ event log
                                                          │
                 SQLite + CSV + JSON + snapshots + heatmap + video
```

---

## Quick start

```bash
pip install -r requirements.txt
```

`python run.py` with no arguments follows `configs/app.yaml`: it uses the camera
if `CLASSROOM_RTSP_URL` is set (in the environment or in a `.env` file), and
falls back to the bundled demo recording if it is not. So be explicit about
which one you want.

Run the demo recording:

```bash
python run.py --source ../11-classroom-monitoring/demo/class_videos/camera_test.mp4
```

Run your own recording, without a preview window:

```bash
python run.py --source class_videos/lesson.mp4 --no-view
```

Run the live camera. Put the URL in a `.env` file (copy `.env.example`) so the
password never reaches a commit, and check the stream first:

```bash
python tools/camera_check.py
```

```bash
python run.py --source env:CLASSROOM_RTSP_URL --preset realtime
```

A live source never ends — stop it with `q` in the preview window or `Ctrl+C`,
and the session is finalised properly on the way out.

Then read the results back:

```bash
python tools/report.py --charts
```

While the preview window is open: `q` quits, `space` pauses, `h` hides the
overlay.

---

## What comes out

Every run creates `outputs/session_<timestamp>/`:

| File | What it is |
| --- | --- |
| `annotated.mp4` | the video with boxes, IDs, trails, zones and the live HUD |
| `tracks.csv` | one row per student per frame — position, velocity, zone, movement |
| `session.json` | full summary: students, events, trajectories, occupancy, config used |
| `occupancy.csv` | how many people were in the room, bucket by bucket |
| `heatmap.jpg` | where people spent their time, blended over the room |
| `students/student_07/snapshot.jpg` | the best crop of each student |

and appends to **`outputs/analytics.db`**, one SQLite database shared by every
session. That is the difference between "a video with boxes on it" and
something you can ask questions of a month later:

```sql
-- attendance per session, last 30 days
SELECT session_id, started_at, students_detected, peak_present
FROM sessions ORDER BY started_at DESC;

-- who was in the room longest, and how much did they move?
SELECT student_id, duration_s, moving_s, home_zone
FROM students WHERE session_id = 'session_20260820-171530'
ORDER BY duration_s DESC;

-- busiest part of the room across every recording
SELECT zone, SUM(duration_s) AS seconds FROM zone_visits
GROUP BY zone ORDER BY seconds DESC;
```

Tables: `sessions`, `students`, `track_samples`, `events`, `zone_visits`,
`occupancy`.

---

## How the pipeline is built

| Module | Job | Roadmap phase |
| --- | --- | --- |
| `video_source.py` | file / RTSP / webcam, reconnect, newest-frame-wins | 4 — video & camera |
| `preprocess.py` | CLAHE contrast, denoise (optional) | 2 — image processing |
| `detector.py` | YOLO11-nano person detection + BoT-SORT + quality gates | 8 — detection, 9 — tracking |
| `kalman.py` | constant-velocity Kalman filter over the box | 9 — Kalman, prediction, correction |
| `registry.py` | stable student IDs, occlusion coasting, re-identification | 9 — track management |
| `zones.py` | classroom zones and ROI polygons | 3 — contours, ROI |
| `motion.py` | MOG2 background subtraction → movement/activity | 5 — motion detection |
| `analytics.py` | occupancy timeline, heatmap, per-student statistics | 11 — statistics |
| `storage.py` | SQLite + CSV + JSON + snapshots | 11 — event logging |
| `visualizer.py` | boxes, labels, trails, HUD dashboard | 1 — drawing |
| `pipeline.py` | wires it all together | 12 — final demo |

One ordering rule matters: **every measurement is taken from the clean frame,
and only a copy is annotated**. Drawing first would feed boxes and labels into
the background subtractor and into the students' appearance histograms.

---

## The three hard requirements

### 1. Don't report people who aren't there

The instinct is to raise the confidence threshold. On this footage that is
exactly wrong. Measured over the demo clip with YOLO11-nano at `imgsz=960`:

| confidence ≥ | boxes per frame |
| --- | --- |
| 0.10 | 11.6 |
| 0.25 | 5.2 |
| 0.40 | 2.5 |
| 0.55 | 1.3 |

There are **six** people in that room. At 0.55 the detector reports barely one
box per frame; the real people are down at 0.20–0.45, mixed in with the noise.
Nano does not produce high confidences on small, seated, partly occluded
people, so no threshold on this axis separates "real student" from "false
positive" — a high one just deletes the back rows.

So confidence is not used as the filter. Five other things are:

* **Persistence.** A raw track becomes "Student N" only after `min_hits`
  frames — 20 by default, one second at 20 fps. Noise does not survive a
  second; people do. Tracks that never make it are counted as
  `tracks_discarded` and never touch the database.
* **Geometry.** Area between 0.04% and 55% of the frame, height/width between
  0.55 and 6.0, shortest side ≥ 18 px. A box that cannot be a person is not one.
* **Duplicates.** YOLO regularly reports a torso box *and* a full-body box for
  one seated student. Their IoU can be as low as 0.3 while one sits ~100%
  inside the other, so IoU alone misses it. Containment
  (intersection ÷ smaller area ≥ 0.80) catches it, and the box belonging to an
  already-established student wins — otherwise the winner flips between frames
  and one person looks like two taking turns. On the demo clip this removes a
  quarter of all boxes: 1118 of 4409.
* **One body, one number.** The mirror image of the above, one level up: a
  track about to be confirmed on top of a student who is already on screen is
  thrown away instead. Without it, a student who flickers while occluded picks
  up a second identity — which is exactly what happened before this check
  existed, and is why the demo clip briefly reported 7 people for 6.
* **ROI.** An optional polygon; anything whose feet land outside it is ignored.
  Useful when a corridor or a neighbouring room is visible through glass.

### 2. Boxes that don't jump

`kalman.py` runs a constant-velocity filter per student:

```
state        [cx, cy, w, h, vcx, vcy, vw, vh]      velocities in px/s
measurement  [cx, cy, w, h]
noise        scaled by box height (DeepSORT convention) — a student at the back
             of the room is allowed less absolute movement than one at the front
```

Three things follow from it:

* **The drawn and stored box is the filter's estimate**, never the raw
  detection. Mean frame-to-frame movement of the same box, measured over 1457
  samples on the demo clip — BoT-SORT already smooths internally, and this
  filter is what runs on top of it:

  | | centre (px) | centre p95 | size (px) | size p95 |
  | --- | --- | --- | --- | --- |
  | raw YOLO detections | 0.79 | 2.62 | 1.99 | 6.57 |
  | BoT-SORT output | 0.58 | 2.08 | 1.48 | 5.17 |
  | **this app's estimate** | **0.43** | **1.55** | **0.68** | **2.41** |

  Position jitter drops by ~46% against raw detections and ~26% against the
  tracker; box-size wobble, the most visible kind, drops by ~66% and ~54%.
* **Outliers are gated out.** Each detection is scored by squared Mahalanobis
  distance against the prediction and rejected above the 95% chi-square value
  for 4 degrees of freedom (9.4877). A detection that lands impossibly far away
  — the classic symptom of an ID swap — does not move the box.
* **Reality still wins.** If the tracker insists on the new position for five
  consecutive frames, the filter re-initialises there. Gating suppresses
  glitches, not genuine movement.

Turn the gate off to see the difference: `python run.py --no-kalman-gate`.

### 3. Don't lose people

Two mechanisms, because there are two different failures.

**Short occlusion — someone walks past.** The tracker returns nothing for that
student. The Kalman filter keeps predicting, the student stays on screen with a
dashed box for up to `max_coast_frames` (45 ≈ 2.2 s), and their identity is
untouched. A student the detector was confident about earns the full budget; a
marginal one gets half.

**Renumbering — Ultralytics issues a brand-new track ID for someone it already
had.** This is what makes a naive pipeline report 60 students in a 40-minute
lesson. Before creating a new student, the registry checks whether this track is
somebody it already knows, and rebinds if three signals agree:

```
score = 0.5 · appearance  +  0.3 · proximity  +  0.2 · IoU
```

Appearance is an HSV histogram of the *torso* — the part of a seated person that
stays visible and doesn't change when they turn their head — smoothed over time.
Proximity is measured against the Kalman-predicted position, with a search
radius of 2.2 × box height.

Two kinds of candidate are considered, with different bars:

* **lost** — gone from the screen within the last `max_gap_s` (25 s). Scored on
  all three signals.
* **coasting** — still being predicted right now. The new box must land on that
  prediction (`coasting_iou`, 0.35) before it is even scored, because a
  coasting student is still on screen and a passer-by must not be able to
  inherit their identity.

The coasting case is the one that matters most in practice: on the demo clip it
is what keeps a flickering student from acquiring a second number. Rebinds are
logged as `reidentified` events and counted per student, so you can see exactly
how often it fires — four times on the demo clip, for two people.

Below the registry, `configs/botsort_reid.yaml` also does its part:
`track_buffer: 150` (≈7 s of tracker-level memory) and `with_reid: True`
(appearance matching, the main defence against two students swapping IDs when
they cross).

---

## Measured on the demo clip

`camera_test.mp4` — 601 frames, 1920×1080, 20 fps. Six people are at desks in
the room; five are visible from the first frame and the sixth only becomes
detectable at t≈15 s. More people are visible through glass in a separate room
behind, and should *not* be counted.

| | result |
| --- | --- |
| students reported | **6** — one per person, no splits, nobody through the glass |
| peak present | 6 |
| noise tracks discarded before ever getting a number | 14 |
| duplicate boxes suppressed | 1118 of 4409 raw boxes (25%) |
| tracking quality (frames measured ÷ frames held) | 1.00, 1.00, 0.96, 0.95, 1.00, 0.73 |
| re-identifications | 4 (two students renumbered by the tracker, both reclaimed) |
| processing speed | 6.1 fps on CPU at `imgsz=960` |

Five of the six are held for the entire 30 s clip; the sixth for the 14 s they
are detectable. Every student has a snapshot, so each identity can be checked
by eye — which is how the numbers above were verified, not by trusting the
count.

Reproduce it with:

```bash
python run.py --no-view --source ../11-classroom-monitoring/demo/class_videos/camera_test.mp4 --session demo_camera01
```

For contrast, on the first 300 frames of the same clip: stricter thresholds
(`new_track_thresh: 0.55`, `confirm_conf: 0.40`) reported **3** students, and
relaxed thresholds without containment suppression reported **8**. Each number
is wrong in exactly the way the corresponding gate exists to prevent.

The live path is exercised too: an 8-minute run against the RTSP camera held
eight identities while the camera delivered ~25 fps and the pipeline processed
~5 fps, with the newest-frame-wins reader discarding the rest so the analysis
stayed current instead of falling behind.

---

## Configuration

Everything lives in `configs/app.yaml`, and every key mirrors a dataclass field
in `classroom_analytics/config.py` where the reasoning is written down.

Precedence: **built-in defaults → app.yaml → command-line flags.**

### Tuning cheat sheet

| Symptom | Knob | Direction |
| --- | --- | --- |
| Students in the back row are missed | `detect.imgsz` | up (960 → 1280) |
| Same, and imgsz is maxed | `detect.conf`, `track_high_thresh` | down |
| Phantom students appear briefly | `track.min_hits` | up |
| One person counted as two | `detect.containment_thresh` | down (0.80 → 0.70) |
| Two people merged into one | `detect.containment_thresh`, `detect.iou` | up |
| Boxes still twitch | `kalman.std_position`, `kalman.size_ema` | down |
| Boxes lag behind fast movement | `kalman.std_position`, `kalman.size_ema` | up |
| Students vanish behind a classmate | `track.max_coast_frames`, `track_buffer` | up |
| IDs keep incrementing on the same people | `reid.max_gap_s` up, `reid.min_score` down | |
| Two students swapped identities | `reid.min_appearance`, `appearance_thresh` | up |
| People through a window are counted | `zones.roi` | draw one |
| Too slow | `detect.imgsz` 640, `source.frame_stride` 2, `gmc_method: none` | |

`min_hits` is in **frames**, so scale it with your camera: 20 frames is one
second at 20 fps, but only 0.66 s at 30 fps.

### Zones and ROI

Zones default to a 3×3 grid (`front-left` … `back-right`). To draw real ones —
a teacher's desk, a door, an ROI that excludes the corridor — click them:

```bash
python tools/draw_zones.py --source ../11-classroom-monitoring/demo/class_videos/camera_test.mp4
```

Left-click adds points, `n` names the polygon, `s` writes a YAML snippet you
paste into `app.yaml`. Name one `roi` to make it the region of interest.
Coordinates are normalized (0–1), so a config survives a resolution change.

---

## Command line

```
--source PATH|URL|INDEX|env:VAR   video file, rtsp:// URL, webcam index, env var
--preset quality|realtime         960px accurate, or 640px + sampling for live
--model / --imgsz / --conf        detector
--tracker configs/bytetrack.yaml  swap the tracker
--min-hits / --confirm-conf       how strict "this is a student" is
--coast-frames                    how long an occluded student is held
--stride / --start-frame / --max-frames
--no-view --no-video --no-db --no-snapshots
--no-zones --no-motion --no-reid --no-kalman-gate    ablations
--clahe                           contrast boost for a dim room
--out / --session / --config
```

The `--no-*` flags exist to be used: turning one off and re-running is the
fastest way to see what that stage was actually doing.

```bash
python run.py --no-view --source ../11-classroom-monitoring/demo/class_videos/camera_test.mp4 --session with_reid
python run.py --no-view --no-reid --source ../11-classroom-monitoring/demo/class_videos/camera_test.mp4 --session without_reid
python tools/report.py --compare with_reid without_reid
```

---

## Tools

| Tool | Purpose |
| --- | --- |
| `tools/report.py` | attendance, zones, movement, occupancy and events from the database |
| `tools/camera_check.py` | is the camera reachable, and what is its *real* frame rate? |
| `tools/draw_zones.py` | click zones and ROI polygons on a frame |

```bash
python tools/report.py --list           # every session ever recorded
python tools/report.py --charts         # + PNG charts
python tools/report.py --session demo_camera01
```

---

## Live camera notes

* Put the URL in `.env` as `CLASSROOM_RTSP_URL`. `.env` is gitignored; the URL
  contains a password and must never be committed.
* Hikvision: `/Streaming/Channels/101` is the main stream, `/102` the sub
  stream. The sub stream is far lighter and usually enough at `imgsz=640`.
* RTSP is forced over TCP. UDP drops packets and produces smeared frames, which
  the detector reads as low-confidence garbage.
* Live sources are read in a background thread that keeps only the newest
  frame, plus `CAP_PROP_BUFFERSIZE=1`. Without both, a pipeline slower than the
  camera drifts minutes behind reality.
* Dropped connections are retried (`reconnect_attempts`, `reconnect_delay_s`).
* `Ctrl+C` finalises the session properly — the database, JSON and CSV are
  written for the part that ran.

---

## Performance

Measured on this machine (CPU-only PyTorch, 1080p input):

| Setting | Speed |
| --- | --- |
| `imgsz=960`, BoT-SORT + ReID | ~6 fps |
| `imgsz=640` (`--preset realtime`) | ~2× faster |
| `--stride 2` | halves the work again |

A CUDA GPU changes the picture entirely — set `detect.device: "0"`. For live
monitoring on CPU, `--preset realtime --stride 2 --no-video` on a sub-stream is
the combination that keeps up.

---

## Limitations

* **Identities are not names.** "Student 07" is a tracking identity for one
  session. Nobody is recognised across sessions, and no face recognition is
  performed anywhere in this project.
* **Re-identification uses colour.** Two people in similar clothing who swap
  places while both are occluded can still be confused.
* **`min_hits` costs a second.** Someone who walks through the frame in under a
  second is deliberately not counted.
* **Movement, not behaviour.** The app reports where people were and whether
  they moved. It does not classify what they were doing, and the numbers it
  produces should not be treated as a measure of anyone's attention or effort.
* **Snapshots are photographs of people.** `outputs/` is gitignored by default.
  Keep it that way, and check what your institution requires before recording.

---

## Roadmap coverage

| Phase | Used here |
| --- | --- |
| 1 — OpenCV fundamentals | drawing, colour spaces, HSV histograms |
| 2 — Image processing | CLAHE, morphological opening, thresholding |
| 3 — Contours & ROI | zone polygons, `pointPolygonTest`, ROI masking |
| 4 — Video & camera | capture, FPS, `VideoWriter`, live streams, reconnect |
| 5 — Motion detection | MOG2 background subtraction, shadow removal, activity |
| 6 — Features & matching | HSV histogram descriptors for re-identification |
| 7 — Optical flow | `gmc_method` in the tracker config (off for a fixed camera) |
| 8 — Object detection | YOLO11-nano, confidence, IoU, NMS, containment |
| 9 — Object tracking | BoT-SORT + ReID, Kalman predict/correct, track management |
| 10 — Coursera course | Kalman filtering and multi-object tracking concepts |
| 11 — Classroom monitoring | zones, statistics, snapshots, event logging |
| 12 — Final demo | this application |
