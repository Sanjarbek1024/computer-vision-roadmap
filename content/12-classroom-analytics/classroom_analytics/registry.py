"""Student registry: stable identities on top of the raw tracker (Phase 9).

This is where the three quality requirements are actually met.

**No low-probability students.** A raw tracker ID is not a student. It becomes
one only after `min_hits` frames with a smoothed confidence above
`confirm_conf`. Flickering blobs never get a number, never get drawn, and never
reach the analytics database.

**No jumping boxes.** Every student owns a Kalman filter. The box that gets
drawn and stored is the filter's estimate, and a detection that fails the
chi-square gate is rejected rather than allowed to teleport the box. If the
tracker insists on the new position for several frames in a row, the filter
re-initialises there - reality wins eventually, just not instantly.

**No lost students.** Two mechanisms:

    coasting - no detection this frame? The filter keeps predicting and the
               student stays on screen (drawn dashed) for max_coast_frames.
    re-ID    - Ultralytics hands out a *new* raw ID after a long occlusion.
               Before creating a new student, we check students who are lost
               *or* currently being coasted, and rebind if position and
               appearance agree. Without this, a 40-minute lesson with 12
               students reports 60 of them.

And the mirror image of that: a track that would confirm on top of a student who
is already on screen is thrown away instead, so one person can never end up
holding two student numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AppConfig
from .detector import Detection
from .kalman import BoxKalman, containment, iou

# Distinct, readable on both dark and light classroom footage.
PALETTE = [
    (66, 135, 245), (52, 199, 89), (255, 149, 0), (255, 45, 85),
    (175, 82, 222), (90, 200, 250), (255, 214, 10), (162, 132, 94),
    (48, 209, 88), (255, 105, 97), (100, 210, 255), (191, 90, 242),
]

TENTATIVE = "tentative"
ACTIVE = "active"
COASTING = "coasting"
LOST = "lost"
EXITED = "exited"


@dataclass
class ZoneVisit:
    zone: str
    start_time: float
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class Event:
    frame: int
    time: float
    student_id: int
    type: str
    detail: str = ""


@dataclass
class Student:
    """One person, tracked for as long as the room can hold on to them."""

    internal_id: int
    raw_id: int
    kalman: BoxKalman
    first_frame: int
    first_time: float

    student_id: int = 0          # 0 until confirmed; then the display number
    state: str = TENTATIVE

    hits: int = 0
    misses: int = 0              # consecutive frames without a detection
    visible_frames: int = 0
    coasted_frames: int = 0
    consecutive_outliers: int = 0

    conf: float = 0.0
    conf_ema: float = 0.0
    max_conf: float = 0.0

    last_frame: int = 0
    last_time: float = 0.0
    last_seen_frame: int = 0     # last frame with a real detection
    last_seen_time: float = 0.0

    raw_ids: list[int] = field(default_factory=list)
    rebinds: int = 0

    trajectory: list[tuple[int, float, int, int]] = field(default_factory=list)
    distance_px: float = 0.0
    _last_point: tuple[float, float] | None = None

    zone: str = "unassigned"
    zone_visits: list[ZoneVisit] = field(default_factory=list)
    zone_time: dict[str, float] = field(default_factory=dict)
    _pending_zone: str = ""
    _pending_zone_count: int = 0

    moving: bool = False
    moving_seconds: float = 0.0
    _moving_count: int = 0
    _still_count: int = 0
    activity: float = 0.0

    appearance: np.ndarray | None = None
    snapshot: np.ndarray | None = None
    snapshot_score: float = 0.0
    snapshot_frame: int = 0
    snapshot_path: str = ""

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.kalman.box

    @property
    def center(self) -> tuple[float, float]:
        return self.kalman.center

    @property
    def anchor(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def color(self) -> tuple[int, int, int]:
        key = self.student_id or self.internal_id
        return PALETTE[key % len(PALETTE)]

    @property
    def label(self) -> str:
        return f"S{self.student_id:02d}" if self.student_id else "?"

    @property
    def confirmed(self) -> bool:
        return self.student_id > 0

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen_time - self.first_time)

    @property
    def home_zone(self) -> str:
        if not self.zone_time:
            return self.zone
        return max(self.zone_time.items(), key=lambda item: item[1])[0]


class StudentRegistry:
    """Owns every student, alive or lost, for the whole session."""

    def __init__(self, cfg: AppConfig, fps: float):
        self.cfg = cfg
        self.fps = fps or 25.0

        self.students: dict[int, Student] = {}     # internal_id -> Student
        self.by_raw_id: dict[int, int] = {}        # raw tracker ID -> internal_id
        self.events: list[Event] = []

        self._next_internal = 1
        self._next_student_id = 1
        self._discarded_tracks = 0                 # never reached confirmation

    # ------------------------------------------------------------------ API
    @property
    def confirmed_students(self) -> list[Student]:
        return sorted(
            (s for s in self.students.values() if s.confirmed),
            key=lambda s: s.student_id,
        )

    @property
    def total_confirmed(self) -> int:
        return self._next_student_id - 1

    @property
    def discarded_tracks(self) -> int:
        return self._discarded_tracks

    def visible(self) -> list[Student]:
        """Students to draw and record this frame."""
        return [
            s for s in self.students.values()
            if s.confirmed and s.state in (ACTIVE, COASTING)
        ]

    def is_established(self, raw_id: int) -> bool:
        """Does this raw tracker ID already belong to a confirmed student?

        Used by the duplicate gate: when two boxes overlap, the one that is
        already somebody wins, so the winner does not flip frame to frame.
        """
        internal_id = self.by_raw_id.get(int(raw_id))
        if internal_id is None:
            return False
        student = self.students.get(internal_id)
        return bool(student and student.confirmed
                    and student.state in (ACTIVE, COASTING))

    def present_count(self) -> int:
        return sum(1 for s in self.students.values()
                   if s.confirmed and s.state == ACTIVE)

    # --------------------------------------------------------------- update
    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        dt: float,
        zone_map=None,
        motion=None,
    ) -> list[Student]:
        """Advance every student by one frame and fold in this frame's detections."""
        track_cfg = self.cfg.track

        # 1. PREDICT for everyone still in play. This has to happen before
        #    matching: the gate and the re-ID search both use the prediction.
        live = [s for s in self.students.values()
                if s.state in (TENTATIVE, ACTIVE, COASTING)]
        for student in live:
            student.kalman.predict(dt)

        # 2. Detections whose raw ID we already know go straight to their owner.
        matched: dict[int, Detection] = {}
        unclaimed: list[Detection] = []

        for detection in detections:
            internal_id = self.by_raw_id.get(detection.track_id)
            student = self.students.get(internal_id) if internal_id else None

            if student is not None and student.state in (TENTATIVE, ACTIVE, COASTING):
                # Two detections claiming one student (rare): keep the confident one.
                previous = matched.get(student.internal_id)
                if previous is None or detection.conf > previous.conf:
                    if previous is not None:
                        unclaimed.append(previous)
                    matched[student.internal_id] = detection
                else:
                    unclaimed.append(detection)
            else:
                unclaimed.append(detection)

        # 3. New raw IDs: rebind to a recently lost student, or start a new one.
        for detection in unclaimed:
            student = self._rebind(detection, frame, timestamp)

            if student is None:
                student = self._create(detection, frame, frame_idx, timestamp)
            else:
                student.rebinds += 1
                self._log(frame_idx, timestamp, student, "reidentified",
                          f"raw_id {detection.track_id}")

            self.by_raw_id[detection.track_id] = student.internal_id
            matched[student.internal_id] = detection

        # 4. Fold the measurements in.
        for internal_id, detection in matched.items():
            self._apply_detection(
                self.students[internal_id], detection, frame,
                frame_idx, timestamp, dt, zone_map, motion,
            )

        # 5. Everyone else coasts on their prediction.
        for student in live:
            if student.internal_id in matched:
                continue
            self._apply_miss(student, frame_idx, timestamp, dt, zone_map)

        # 6. Retire students who have been lost long enough to call it a day.
        for student in list(self.students.values()):
            if student.state == LOST and \
                    timestamp - student.last_seen_time > track_cfg.forget_after_s:
                self._retire(student, frame_idx, timestamp)

        return self.visible()

    # ------------------------------------------------------- matched student
    def _apply_detection(
        self,
        student: Student,
        detection: Detection,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        dt: float,
        zone_map,
        motion,
    ) -> None:
        track_cfg = self.cfg.track

        accepted, _distance = student.kalman.correct(detection.box)

        if accepted:
            student.consecutive_outliers = 0
        else:
            # The detection was gated out as implausible. Tolerate a short burst
            # (occlusion artefacts), but if the tracker keeps insisting, believe
            # it and restart the filter there.
            student.consecutive_outliers += 1
            if student.consecutive_outliers >= 5:
                student.kalman = BoxKalman(detection.box, self.cfg.kalman)
                student.consecutive_outliers = 0

        if detection.track_id not in student.raw_ids:
            student.raw_ids.append(detection.track_id)

        student.hits += 1
        student.misses = 0
        student.visible_frames += 1
        student.conf = detection.conf
        student.max_conf = max(student.max_conf, detection.conf)

        alpha = track_cfg.conf_ema
        student.conf_ema = (1 - alpha) * student.conf_ema + alpha * detection.conf

        student.last_frame = frame_idx
        student.last_time = timestamp
        student.last_seen_frame = frame_idx
        student.last_seen_time = timestamp

        self._update_appearance(student, frame, detection.box)
        self._update_snapshot(student, frame, detection, frame_idx)

        # Promotion: a raw track becomes a numbered student here, and nowhere else.
        if student.state == TENTATIVE:
            if student.hits >= track_cfg.min_hits and \
                    student.conf_ema >= track_cfg.confirm_conf:
                if self._sits_on_existing_student(student):
                    # Someone is already being tracked on this body. Confirming
                    # would split one person into two students, so this track is
                    # thrown away instead; the older identity keeps the person.
                    self._discard(student)
                    return

                student.student_id = self._next_student_id
                self._next_student_id += 1
                student.state = ACTIVE
                self._log(frame_idx, timestamp, student, "entered",
                          f"conf={student.conf_ema:.2f}")
        else:
            if student.state == COASTING:
                self._log(frame_idx, timestamp, student, "recovered",
                          f"after {student.coasted_frames} predicted frames")
            student.state = ACTIVE

        self._update_motion(student, frame_idx, timestamp, dt, zone_map, motion)

    # --------------------------------------------------------- missed student
    def _apply_miss(
        self, student: Student, frame_idx: int, timestamp: float, dt: float, zone_map
    ) -> None:
        track_cfg = self.cfg.track

        student.misses += 1
        student.last_frame = frame_idx
        student.last_time = timestamp

        if student.state == TENTATIVE:
            # An unconfirmed track that stops producing detections was noise.
            if student.misses > max(3, track_cfg.min_hits // 2):
                self._discard(student)
            return

        student.coasted_frames += 1

        if student.state == ACTIVE:
            student.state = COASTING
            self._log(frame_idx, timestamp, student, "occluded", "")

        # How long we hold someone depends on how solid they were while
        # visible: a student the detector was confident about earns the full
        # occlusion budget, a marginal one gets half of it. `conf_ema` is
        # deliberately NOT decayed here - it stays a measure of detection
        # quality, which is what the reports display.
        budget = track_cfg.max_coast_frames
        if student.conf_ema < track_cfg.drop_conf:
            budget //= 2

        if student.misses > budget:
            student.state = LOST
            self._close_zone_visit(student, timestamp)
            self._log(frame_idx, timestamp, student, "lost", "")
            # Free the raw IDs so a future detection with the same number is
            # treated as a fresh candidate rather than silently inheriting this
            # student.
            for raw_id in student.raw_ids:
                if self.by_raw_id.get(raw_id) == student.internal_id:
                    self.by_raw_id.pop(raw_id, None)

    def _sits_on_existing_student(self, candidate: Student) -> bool:
        """Is this track covering a body that already belongs to somebody?"""
        threshold = self.cfg.track.duplicate_student_iou
        containment_threshold = self.cfg.detect.containment_thresh

        for student in self.students.values():
            if student.internal_id == candidate.internal_id or not student.confirmed:
                continue
            if student.state not in (ACTIVE, COASTING):
                continue
            if iou(candidate.box, student.box) >= threshold or \
                    containment(candidate.box, student.box) >= containment_threshold:
                return True
        return False

    # ------------------------------------------------------------ life cycle
    def _create(
        self, detection: Detection, frame: np.ndarray, frame_idx: int, timestamp: float
    ) -> Student:
        student = Student(
            internal_id=self._next_internal,
            raw_id=detection.track_id,
            kalman=BoxKalman(detection.box, self.cfg.kalman),
            first_frame=frame_idx,
            first_time=timestamp,
        )
        student.raw_ids.append(detection.track_id)
        student.conf_ema = detection.conf
        self._next_internal += 1
        self.students[student.internal_id] = student
        return student

    def _discard(self, student: Student) -> None:
        """Drop an unconfirmed track without consuming a student number."""
        for raw_id in student.raw_ids:
            if self.by_raw_id.get(raw_id) == student.internal_id:
                self.by_raw_id.pop(raw_id, None)
        self.students.pop(student.internal_id, None)
        self._discarded_tracks += 1

    def _retire(self, student: Student, frame_idx: int, timestamp: float) -> None:
        student.state = EXITED
        self._close_zone_visit(student, student.last_seen_time)
        self._log(frame_idx, timestamp, student, "left",
                  f"seen {student.visible_frames} frames")

    def finalize(self, frame_idx: int, timestamp: float) -> None:
        """Close open zone visits at end of session so durations are complete."""
        for student in self.students.values():
            if student.state in (TENTATIVE,):
                continue
            if student.state != EXITED:
                self._close_zone_visit(student, student.last_seen_time or timestamp)
                student.state = EXITED

    # ------------------------------------------------------------------ ReID
    def _rebind(
        self, detection: Detection, frame: np.ndarray, timestamp: float
    ) -> Student | None:
        """Find the student this brand-new track probably already is.

        Two kinds of candidate, with different bars:

            lost     - gone from the screen recently; matched on appearance,
                       proximity to the last prediction, and overlap.
            coasting - still being predicted right now. The tracker simply
                       renumbered someone it never actually lost, so the new box
                       has to land on the prediction (`coasting_iou`) before we
                       will even score it.
        """
        cfg = self.cfg.reid
        if not cfg.enabled:
            return None

        descriptor = self._descriptor(frame, detection.box)
        cx, cy = detection.center
        radius = max(60.0, cfg.max_distance_scale * detection.height)

        best: tuple[float, Student] | None = None

        for student in self.students.values():
            if not student.confirmed:
                continue
            if timestamp - student.last_seen_time > cfg.max_gap_s:
                continue

            if student.state == COASTING:
                # Still on screen as a prediction. Only reclaim them if the new
                # box lands on that prediction - anything looser would let a
                # passer-by inherit an occluded student's identity.
                if iou(detection.box, student.box) < cfg.coasting_iou:
                    continue
            elif student.state != LOST:
                continue

            sx, sy = student.center
            distance = float(np.hypot(cx - sx, cy - sy))
            if distance > radius:
                continue

            appearance = 0.0
            if descriptor is not None and student.appearance is not None:
                appearance = float(
                    cv2.compareHist(student.appearance, descriptor, cv2.HISTCMP_CORREL)
                )
                if appearance < cfg.min_appearance:
                    continue
            elif descriptor is not None or student.appearance is not None:
                # One side has no usable crop (student was at the frame edge).
                # Fall back to geometry alone, at a penalty.
                appearance = cfg.min_appearance

            overlap = iou(detection.box, student.box)

            score = (
                cfg.w_appearance * max(0.0, appearance)
                + cfg.w_distance * (1.0 - distance / radius)
                + cfg.w_iou * overlap
            )

            if score >= cfg.min_score and (best is None or score > best[0]):
                best = (score, student)

        if best is None:
            return None

        student = best[1]

        # The tracker gave this person a new number; drop the old mappings so a
        # stale raw ID cannot later feed a second detection into this student.
        for raw_id in student.raw_ids:
            if self.by_raw_id.get(raw_id) == student.internal_id:
                self.by_raw_id.pop(raw_id, None)

        student.state = ACTIVE
        student.misses = 0
        student.consecutive_outliers = 0
        # Restart the filter from the new observation: after a long gap the old
        # velocity estimate is meaningless.
        student.kalman = BoxKalman(detection.box, self.cfg.kalman)
        return student

    def _descriptor(self, frame: np.ndarray, box) -> np.ndarray | None:
        """HSV colour histogram of the torso - the part of a seated student
        that stays visible and does not change with head movement."""
        cfg = self.cfg.reid
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box

        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            return None

        tx1 = int(max(0, x1 + 0.20 * bw))
        tx2 = int(min(w, x2 - 0.20 * bw))
        ty1 = int(max(0, y1 + 0.15 * bh))
        ty2 = int(min(h, y1 + 0.60 * bh))

        if tx2 - tx1 < 4 or ty2 - ty1 < 4:
            return None

        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, list(cfg.hist_bins), [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
        return hist.astype(np.float32)

    def _update_appearance(self, student: Student, frame: np.ndarray, box) -> None:
        descriptor = self._descriptor(frame, box)
        if descriptor is None:
            return
        if student.appearance is None:
            student.appearance = descriptor
            return
        alpha = self.cfg.reid.hist_ema
        student.appearance = ((1 - alpha) * student.appearance
                              + alpha * descriptor).astype(np.float32)

    # -------------------------------------------------------------- snapshot
    def _update_snapshot(
        self, student: Student, frame: np.ndarray, detection: Detection, frame_idx: int
    ) -> None:
        """Keep the single best crop of each student for the report.

        "Best" balances three things: detector confidence, how large the student
        is in frame, and how sharp the crop is (a Laplacian variance, so a
        motion-blurred frame loses to a still one).

        Note there is no hard confidence cut here. With a nano model, a student
        in a back row may never produce a confident box, and "no picture at all"
        is a worse outcome than "the best picture we got". `snapshot_min_conf`
        is a preference instead: crops above it score far higher, so a confident
        one always wins when one exists.
        """
        if not self.cfg.output.save_snapshots:
            return

        h, w = frame.shape[:2]
        x1 = int(max(0, detection.box[0]))
        y1 = int(max(0, detection.box[1]))
        x2 = int(min(w, detection.box[2]))
        y2 = int(min(h, detection.box[3]))
        if x2 - x1 < 12 or y2 - y1 < 24:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        area_frac = ((x2 - x1) * (y2 - y1)) / float(w * h)
        score = (
            detection.conf
            * min(1.0, sharpness / 150.0)
            * min(1.0, area_frac / 0.02)
        )

        if detection.conf >= self.cfg.output.snapshot_min_conf:
            score *= 3.0

        if score > student.snapshot_score:
            student.snapshot_score = score
            student.snapshot = crop.copy()
            student.snapshot_frame = frame_idx

    # ------------------------------------------------------- zones + movement
    def _update_motion(
        self,
        student: Student,
        frame_idx: int,
        timestamp: float,
        dt: float,
        zone_map,
        motion,
    ) -> None:
        cfg = self.cfg.motion

        # --- trajectory + distance travelled (Kalman-smoothed centre) --------
        cx, cy = student.center
        if student._last_point is not None:
            step = float(np.hypot(cx - student._last_point[0],
                                  cy - student._last_point[1]))
            # Ignore sub-pixel wobble so a still student does not accumulate
            # hundreds of "travelled" pixels over a lesson.
            if step > 1.5:
                student.distance_px += step
        student._last_point = (cx, cy)

        every = max(1, self.cfg.output.trajectory_every)
        if student.confirmed and frame_idx % every == 0:
            student.trajectory.append((frame_idx, timestamp, int(cx), int(cy)))

        # --- zone membership, debounced -------------------------------------
        if zone_map is not None and zone_map.zones:
            current = zone_map.zone_for(student.anchor)
            if current != student.zone:
                if current == student._pending_zone:
                    student._pending_zone_count += 1
                else:
                    student._pending_zone = current
                    student._pending_zone_count = 1

                if student._pending_zone_count >= 3:
                    previous = student.zone
                    self._close_zone_visit(student, timestamp)
                    student.zone = current
                    student.zone_visits.append(ZoneVisit(current, timestamp))
                    student._pending_zone_count = 0
                    if student.confirmed and previous != "unassigned":
                        self._log(frame_idx, timestamp, student, "zone_change",
                                  f"{previous} -> {current}")
            else:
                student._pending_zone_count = 0
                if not student.zone_visits:
                    student.zone_visits.append(ZoneVisit(current, timestamp))

            student.zone_time[student.zone] = student.zone_time.get(student.zone, 0.0) + dt

        # --- moving vs still, debounced -------------------------------------
        if motion is not None and motion.enabled:
            raw_activity = motion.activity(student.box)
            student.activity = 0.7 * student.activity + 0.3 * raw_activity

        signal = (student.kalman.speed > cfg.speed_px_per_s
                  or student.activity > cfg.activity_frac)

        if signal:
            student._moving_count += 1
            student._still_count = 0
        else:
            student._still_count += 1
            student._moving_count = 0

        if student.moving:
            student.moving_seconds += dt
            if student._still_count >= cfg.min_frames:
                student.moving = False
                if student.confirmed:
                    self._log(frame_idx, timestamp, student, "stopped_moving", "")
        elif student._moving_count >= cfg.min_frames:
            student.moving = True
            if student.confirmed:
                self._log(frame_idx, timestamp, student, "started_moving",
                          f"{student.kalman.speed:.0f} px/s")

    def _close_zone_visit(self, student: Student, timestamp: float) -> None:
        if student.zone_visits and student.zone_visits[-1].end_time == 0.0:
            student.zone_visits[-1].end_time = timestamp

    # ---------------------------------------------------------------- events
    def _log(self, frame_idx: int, timestamp: float, student: Student,
             event_type: str, detail: str) -> None:
        if not student.confirmed:
            return
        self.events.append(
            Event(frame_idx, timestamp, student.student_id, event_type, detail)
        )
