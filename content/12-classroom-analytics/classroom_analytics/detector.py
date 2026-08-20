"""Person detection + multi-object tracking (Phase 8 + Phase 9).

Ultralytics does detection and tracker association in one `model.track()` call.
Everything this module adds around that call exists to answer one requirement:
*do not report things that are not students*.

The gates, in order:

    confidence   - handled by YOLO itself (kept low on purpose, see DetectConfig)
    class        - person only
    geometry     - area, aspect ratio and minimum pixel size
    ROI          - the student's feet must be inside the monitored region
    duplicates   - two boxes on one person: near-identical (IoU) or nested
                   (one contained in the other), which is what YOLO actually
                   produces on seated people - a torso box and a body box

Only detections that survive all of them reach the student registry. Detections
that fail are counted, not silently dropped, so the HUD can show how much noise
the gates are absorbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import AppConfig
from .kalman import containment, iou


@dataclass
class Detection:
    """A tracked person that passed every quality gate."""

    track_id: int                                   # raw ID from BoT-SORT
    box: tuple[float, float, float, float]          # xyxy in frame pixels
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre point ("feet") - what zone membership is judged on."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class GateStats:
    """Per-frame accounting of what the gates threw away."""

    raw: int = 0
    kept: int = 0
    no_track_id: int = 0
    too_small: int = 0
    bad_aspect: int = 0
    outside_roi: int = 0
    duplicate: int = 0

    def merge(self, other: "GateStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def rejected(self) -> int:
        return self.no_track_id + self.too_small + self.bad_aspect + \
            self.outside_roi + self.duplicate


class PersonDetector:
    """YOLO-nano person detection wired to a BoT-SORT tracker."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.detect = cfg.detect
        self.track_cfg = cfg.track

        self.model = None
        self.tracker_path = str(cfg.resolve(self.track_cfg.tracker))
        self.tracker_name = self.track_cfg.tracker
        self._fallback_used = False
        self.stats = GateStats()

    # ------------------------------------------------------------------ load
    def load(self) -> "PersonDetector":
        from ultralytics import YOLO       # imported late: it is a slow import

        model_ref = self.detect.model
        local = self.cfg.resolve(model_ref)
        if local.exists():
            model_ref = str(local)
        else:
            # Not in the project folder - try the repo root, where the other
            # roadmap notebooks keep their weights. Otherwise let Ultralytics
            # download it by name.
            from .config import ROOT_DIR
            root_copy = ROOT_DIR / self.detect.model
            if root_copy.exists():
                model_ref = str(root_copy)

        self.model = YOLO(model_ref)
        return self

    # -------------------------------------------------------------- tracking
    def track(
        self,
        frame: np.ndarray,
        roi_test: Callable[[tuple[float, float]], bool] | None = None,
        established: Callable[[int], bool] | None = None,
    ) -> tuple[list[Detection], GateStats]:
        """Detect + track people in one frame and return the survivors."""
        if self.model is None:
            self.load()

        result = self._run_tracker(frame)
        stats = GateStats()

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return [], stats

        stats.raw = len(boxes)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        if ids is None:
            # The tracker has not assigned IDs yet (very first frames, or every
            # detection fell below track_high_thresh). Nothing to associate.
            stats.no_track_id = len(xyxy)
            return [], stats

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        candidates: list[Detection] = []

        for box, conf, track_id in zip(xyxy, confs, ids):
            x1, y1, x2, y2 = (float(v) for v in box)

            # Clamp to the frame: a box hanging off the edge otherwise reports a
            # width the student does not actually have.
            x1 = max(0.0, min(x1, frame_w - 1.0))
            y1 = max(0.0, min(y1, frame_h - 1.0))
            x2 = max(0.0, min(x2, float(frame_w)))
            y2 = max(0.0, min(y2, float(frame_h)))

            w, h = x2 - x1, y2 - y1
            if w <= 1.0 or h <= 1.0:
                stats.too_small += 1
                continue

            area_frac = (w * h) / frame_area
            if (area_frac < self.detect.min_area_frac
                    or area_frac > self.detect.max_area_frac
                    or min(w, h) < self.detect.min_box_px):
                stats.too_small += 1
                continue

            aspect = h / w
            if not (self.detect.min_aspect <= aspect <= self.detect.max_aspect):
                stats.bad_aspect += 1
                continue

            detection = Detection(int(track_id), (x1, y1, x2, y2), float(conf))

            if roi_test is not None and not roi_test(detection.anchor):
                stats.outside_roi += 1
                continue

            candidates.append(detection)

        kept = self._drop_duplicates(candidates, stats, established)
        stats.kept = len(kept)
        self.stats.merge(stats)
        return kept, stats

    # ------------------------------------------------------------ internals
    def _run_tracker(self, frame: np.ndarray):
        """Call model.track(), falling back to a ReID-free tracker if needed."""
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_path,
                classes=self.detect.classes,
                conf=self.detect.conf,
                iou=self.detect.iou,
                imgsz=self.detect.imgsz,
                max_det=self.detect.max_det,
                device=self.detect.device or None,
                verbose=False,
            )
        except Exception as exc:                       # noqa: BLE001
            if self._fallback_used or not self.track_cfg.fallback_tracker:
                raise
            # Most common cause: the ReID weights could not be downloaded.
            print(f"[detector] tracker '{self.tracker_name}' failed ({exc}); "
                  f"falling back to {self.track_cfg.fallback_tracker}")
            self._fallback_used = True
            self.tracker_name = self.track_cfg.fallback_tracker
            self.tracker_path = str(self.cfg.resolve(self.track_cfg.fallback_tracker))
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                # Ultralytics skips tracker setup when `predictor.trackers`
                # exists and persist=True, so the attribute has to go away
                # entirely for the fallback tracker to be built.
                delattr(predictor, "trackers")
            return self._run_tracker(frame)

        return results[0]

    def _drop_duplicates(
        self,
        detections: list[Detection],
        stats: GateStats,
        established: Callable[[int], bool] | None = None,
    ) -> list[Detection]:
        """Collapse several boxes on one person down to one.

        Two overlap measures, because they catch different failures:

            IoU         - two boxes in the same place at the same scale
            containment - a small box nested inside a large one, which is what
                          YOLO reports for a seated student (torso + body).
                          Their IoU can be 0.3 while one is 100% inside the other.

        Ordering matters as much as the thresholds. Confidence alone makes the
        winner flip between frames when two boxes are close, and a flipping
        winner looks like two students taking turns. So a box whose track is
        already an established student wins first, and confidence only breaks
        ties among newcomers.

        The IoU threshold stays high (0.85): students sitting shoulder to
        shoulder overlap a lot, and merging two real people is a worse error
        than keeping one duplicate.
        """
        if len(detections) < 2:
            return detections

        def rank(detection: Detection) -> tuple[int, float]:
            settled = 1 if (established and established(detection.track_id)) else 0
            return (settled, detection.conf)

        ordered = sorted(detections, key=rank, reverse=True)
        kept: list[Detection] = []

        for detection in ordered:
            duplicate = any(
                iou(detection.box, other.box) >= self.detect.duplicate_iou
                or containment(detection.box, other.box) >= self.detect.containment_thresh
                for other in kept
            )
            if duplicate:
                stats.duplicate += 1
                continue
            kept.append(detection)

        return kept

    def describe(self) -> str:
        return (f"{self.detect.model} | imgsz={self.detect.imgsz} "
                f"conf={self.detect.conf} iou={self.detect.iou} | "
                f"tracker={self.tracker_name}")
