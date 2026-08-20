"""Movement and activity from background subtraction (Phase 5).

The Kalman filter already tells us how fast a student's *box* is moving, but a
student who leans over, raises a hand or turns around barely moves their box at
all. Background subtraction catches that: it measures how many pixels inside
the box changed relative to the learned background of the room.

The two signals are complementary and the pipeline uses both:

    Kalman speed   -> the student is walking across the room
    MOG2 activity  -> the student is moving in place

MOG2 runs on a downscaled grayscale frame. At 0.4 scale that is ~6x less work,
and the signal we want (a fraction of changed pixels per box) survives the
resize fine.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import MotionConfig


class MotionAnalyzer:
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self.enabled = cfg.enabled

        self._subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=cfg.history,
                varThreshold=cfg.var_threshold,
                detectShadows=cfg.detect_shadows,
            )
            if cfg.enabled
            else None
        )

        ksize = max(1, cfg.open_kernel)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        self.mask: np.ndarray | None = None
        self._scale = max(0.1, min(1.0, cfg.scale))

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """Learn the background and return the cleaned foreground mask."""
        if self._subtractor is None:
            return None

        small = cv2.resize(frame, None, fx=self._scale, fy=self._scale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        mask = self._subtractor.apply(gray)

        # MOG2 marks shadows as 127. Shadows are not movement, so drop them.
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Opening = erode then dilate: removes isolated speckle without eating
        # into the real blobs (Phase 2 morphology).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        self.mask = mask
        return mask

    def activity(self, box: tuple[float, float, float, float]) -> float:
        """Fraction of foreground pixels inside a box, 0..1."""
        if self.mask is None:
            return 0.0

        h, w = self.mask.shape[:2]
        x1 = int(max(0, min(box[0] * self._scale, w - 1)))
        y1 = int(max(0, min(box[1] * self._scale, h - 1)))
        x2 = int(max(0, min(box[2] * self._scale, w)))
        y2 = int(max(0, min(box[3] * self._scale, h)))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        region = self.mask[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        return float(np.count_nonzero(region)) / float(region.size)
