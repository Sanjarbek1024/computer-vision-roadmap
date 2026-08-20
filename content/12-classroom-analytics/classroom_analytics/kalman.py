"""Kalman filtering for a bounding box (Phase 9: predict -> measure -> correct).

BoT-SORT already runs a Kalman filter internally to associate detections with
tracks. This is a *second*, independent filter that sits on top of the tracker
output and does three jobs the tracker does not do for us:

1. Smoothing      - the drawn box is the filter's estimate, not the raw
                    detection, so boxes stop twitching frame to frame.
2. Outlier gating - a detection that lands impossibly far from the prediction
                    (a classic ID-swap symptom) is rejected instead of teleporting
                    the box across the room.
3. Coasting       - when a student is occluded and there is no measurement at
                    all, the filter keeps predicting where they are, so we can
                    hold their identity and keep drawing them.

State vector (8):   [cx, cy, w, h, vcx, vcy, vw, vh]
Measurement (4):    [cx, cy, w, h]
Motion model:       constant velocity, dt in seconds -> velocities are px/s.

Process and measurement noise are scaled by box height, the DeepSORT
convention: a student far from the camera occupies few pixels, so their
absolute position noise is smaller than a student in the front row.
"""

from __future__ import annotations

import numpy as np

try:                       # OpenCV is required, but keep the import obvious
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenCV is required: pip install opencv-python") from exc

from .config import KalmanConfig


def xyxy_to_cxcywh(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
        dtype=np.float32,
    )


def cxcywh_to_xyxy(state) -> tuple[float, float, float, float]:
    cx, cy, w, h = state[:4]
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def iou(box_a, box_b) -> float:
    """Intersection over Union (Phase 8) - reused by the ReID scorer."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def containment(box_a, box_b) -> float:
    """Intersection over the *smaller* box's area.

    1.0 means one box sits entirely inside the other. Unlike IoU this does not
    shrink when the two boxes differ wildly in size, which is exactly the case
    for a torso box nested inside a full-body box of the same student.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / min(area_a, area_b))


class BoxKalman:
    """Constant-velocity Kalman filter over a bounding box."""

    def __init__(self, box_xyxy, cfg: KalmanConfig):
        self.cfg = cfg

        measurement = xyxy_to_cxcywh(box_xyxy)

        self.kf = cv2.KalmanFilter(8, 4, 0)

        # --- motion model: x' = F x  (dt is filled in on every predict) ------
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)

        # --- measurement model: z = H x  (we observe position and size) ------
        self.kf.measurementMatrix = np.hstack(
            [np.eye(4, dtype=np.float32), np.zeros((4, 4), dtype=np.float32)]
        )

        # --- initial state: measured box, zero velocity ----------------------
        state = np.zeros((8, 1), dtype=np.float32)
        state[:4, 0] = measurement
        self.kf.statePost = state
        self.kf.statePre = state.copy()

        h = float(measurement[3])
        pos_var = (2.0 * cfg.std_position * h) ** 2
        vel_var = (10.0 * cfg.std_velocity * h) ** 2
        self.kf.errorCovPost = np.diag(
            np.array([pos_var] * 4 + [vel_var] * 4, dtype=np.float32)
        )

        # Smoothed size, kept separately so the drawn box does not "breathe".
        self._w = float(measurement[2])
        self._h = float(measurement[3])

        self.age = 0            # predict() calls
        self.hits = 0           # correct() calls
        self.outliers = 0       # rejected measurements

    # ------------------------------------------------------------------ noise
    def _process_noise(self, h: float) -> np.ndarray:
        pos = (self.cfg.std_position * h) ** 2
        vel = (self.cfg.std_velocity * h) ** 2
        return np.diag(np.array([pos] * 4 + [vel] * 4, dtype=np.float32))

    def _measurement_noise(self, h: float) -> np.ndarray:
        var = (self.cfg.std_measurement * h) ** 2
        return np.diag(np.array([var] * 4, dtype=np.float32))

    # ---------------------------------------------------------------- predict
    def predict(self, dt: float) -> tuple[float, float, float, float]:
        """Advance the state by dt seconds and return the predicted box."""
        dt = float(max(1e-3, dt))

        transition = np.eye(8, dtype=np.float32)
        for i in range(4):
            transition[i, i + 4] = dt
        self.kf.transitionMatrix = transition

        h = max(1.0, float(self.kf.statePost[3, 0]))
        self.kf.processNoiseCov = self._process_noise(h)

        self.kf.predict()
        self.age += 1
        return cxcywh_to_xyxy(self.kf.statePre[:, 0])

    # ------------------------------------------------------------------ gate
    def gating_distance(self, box_xyxy) -> float:
        """Squared Mahalanobis distance between a measurement and the prediction.

        Compared against a chi-square threshold with 4 degrees of freedom, this
        answers "could this detection plausibly be the same object?".
        """
        z = xyxy_to_cxcywh(box_xyxy).reshape(4, 1)
        predicted = self.kf.statePre[:4]
        innovation = z - predicted

        h = max(1.0, float(self.kf.statePre[3, 0]))
        s = self.kf.errorCovPre[:4, :4] + self._measurement_noise(h)

        try:
            solved = np.linalg.solve(s.astype(np.float64), innovation.astype(np.float64))
        except np.linalg.LinAlgError:       # pragma: no cover - singular S
            return 0.0

        return float((innovation.T.astype(np.float64) @ solved).item())

    # --------------------------------------------------------------- correct
    def correct(self, box_xyxy) -> tuple[bool, float]:
        """Fold a detection into the estimate.

        Returns (accepted, gating_distance). A rejected measurement leaves the
        filter coasting on its prediction for this frame.
        """
        distance = self.gating_distance(box_xyxy)

        if self.cfg.reject_outliers and distance > self.cfg.gate_chi2:
            self.outliers += 1
            return False, distance

        measurement = xyxy_to_cxcywh(box_xyxy).reshape(4, 1)
        h = max(1.0, float(measurement[3, 0]))
        self.kf.measurementNoiseCov = self._measurement_noise(h)
        self.kf.correct(measurement)
        self.hits += 1

        # Extra size smoothing on top of the filter: box size is the noisiest
        # part of a detection and the most visible when it jitters.
        alpha = self.cfg.size_ema
        self._w = (1 - alpha) * self._w + alpha * float(self.kf.statePost[2, 0])
        self._h = (1 - alpha) * self._h + alpha * float(self.kf.statePost[3, 0])

        return True, distance

    # ----------------------------------------------------------------- state
    @property
    def box(self) -> tuple[float, float, float, float]:
        """Current smoothed box estimate as xyxy."""
        cx = float(self.kf.statePost[0, 0])
        cy = float(self.kf.statePost[1, 0])
        return cxcywh_to_xyxy((cx, cy, self._w, self._h))

    @property
    def center(self) -> tuple[float, float]:
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimated velocity in pixels per second."""
        return float(self.kf.statePost[4, 0]), float(self.kf.statePost[5, 0])

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    @property
    def height(self) -> float:
        return self._h
