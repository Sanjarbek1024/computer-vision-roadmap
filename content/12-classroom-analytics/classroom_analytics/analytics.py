"""Session-level analytics: occupancy over time, heatmaps, summary statistics.

The registry knows about individual students. This module answers the questions
a school actually asks afterwards:

    How many people were in the room, minute by minute?
    Where did people spend their time?
    Who was present, for how long, and how much did they move?
    When did something happen worth looking at?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AppConfig
from .registry import Student, StudentRegistry


@dataclass
class OccupancyBucket:
    start: float
    samples: int = 0
    total_present: int = 0
    total_moving: int = 0
    peak_present: int = 0

    @property
    def avg_present(self) -> float:
        return self.total_present / self.samples if self.samples else 0.0

    @property
    def avg_moving(self) -> float:
        return self.total_moving / self.samples if self.samples else 0.0


@dataclass
class SessionAnalytics:
    cfg: AppConfig
    width: int
    height: int
    fps: float

    buckets: dict[int, OccupancyBucket] = field(default_factory=dict)
    peak_present: int = 0
    peak_time: float = 0.0
    processed_frames: int = 0
    total_gate_rejections: int = 0

    _heatmap: np.ndarray | None = None
    _heat_scale: float = 0.25
    _background: np.ndarray | None = None
    _recent: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        hh = max(1, int(self.height * self._heat_scale))
        hw = max(1, int(self.width * self._heat_scale))
        self._heatmap = np.zeros((hh, hw), dtype=np.float32)

    # --------------------------------------------------------------- update
    def update(
        self,
        frame: np.ndarray,
        timestamp: float,
        students: list[Student],
        gate_rejections: int = 0,
    ) -> None:
        self.processed_frames += 1
        self.total_gate_rejections += gate_rejections

        present = sum(1 for s in students if s.state == "active")
        moving = sum(1 for s in students if s.moving)

        bucket_size = max(1.0, self.cfg.output.occupancy_bucket_s)
        key = int(timestamp // bucket_size)
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = OccupancyBucket(start=key * bucket_size)
            self.buckets[key] = bucket

        bucket.samples += 1
        bucket.total_present += present
        bucket.total_moving += moving
        bucket.peak_present = max(bucket.peak_present, present)

        if present > self.peak_present:
            self.peak_present = present
            self.peak_time = timestamp

        # Rolling window for the HUD sparkline.
        self._recent.append(present)
        if len(self._recent) > 240:
            self._recent.pop(0)

        # Heatmap accumulates where students *stand*, not where their chest is.
        if self._heatmap is not None:
            for student in students:
                ax, ay = student.anchor
                x = int(ax * self._heat_scale)
                y = int(ay * self._heat_scale)
                if 0 <= x < self._heatmap.shape[1] and 0 <= y < self._heatmap.shape[0]:
                    self._heatmap[y, x] += 1.0

        # One background frame for the heatmap render; the last one is fine and
        # costs a single copy per session.
        self._background = frame

    @property
    def recent_occupancy(self) -> list[int]:
        return self._recent

    # -------------------------------------------------------------- heatmap
    def heatmap_image(self) -> np.ndarray | None:
        """Occupancy heatmap blended over a frame of the room."""
        if self._heatmap is None or self._background is None:
            return None
        if float(self._heatmap.max()) <= 0.0:
            return None

        heat = cv2.GaussianBlur(self._heatmap, (0, 0), sigmaX=6, sigmaY=6)
        # Square root compression: without it a single student who never moves
        # saturates the scale and everyone else disappears.
        heat = np.sqrt(heat)
        heat = heat / float(heat.max())

        heat_u8 = (heat * 255).astype(np.uint8)
        heat_u8 = cv2.resize(heat_u8, (self.width, self.height),
                             interpolation=cv2.INTER_LINEAR)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        # Keep the room visible underneath, and let cold areas stay dark rather
        # than painting the whole frame blue.
        mask = (heat_u8.astype(np.float32) / 255.0)[..., None]
        blended = self._background.astype(np.float32) * (1 - 0.75 * mask) \
            + colored.astype(np.float32) * (0.75 * mask)
        return blended.astype(np.uint8)

    # -------------------------------------------------------------- summary
    def occupancy_rows(self) -> list[dict]:
        rows = []
        for key in sorted(self.buckets):
            bucket = self.buckets[key]
            rows.append({
                "bucket_start_s": round(bucket.start, 2),
                "bucket_end_s": round(bucket.start + self.cfg.output.occupancy_bucket_s, 2),
                "samples": bucket.samples,
                "avg_present": round(bucket.avg_present, 2),
                "peak_present": bucket.peak_present,
                "avg_moving": round(bucket.avg_moving, 2),
            })
        return rows

    def student_rows(self, registry: StudentRegistry) -> list[dict]:
        rows = []
        for student in registry.confirmed_students:
            zone_time = {k: round(v, 2) for k, v in sorted(
                student.zone_time.items(), key=lambda item: item[1], reverse=True)}

            rows.append({
                "student_id": student.student_id,
                "first_frame": student.first_frame,
                "last_frame": student.last_seen_frame,
                "first_seen_s": round(student.first_time, 2),
                "last_seen_s": round(student.last_seen_time, 2),
                "duration_s": round(student.duration, 2),
                "visible_frames": student.visible_frames,
                "predicted_frames": student.coasted_frames,
                "tracking_quality": round(
                    student.visible_frames
                    / max(1, student.visible_frames + student.coasted_frames), 3),
                "avg_confidence": round(student.conf_ema, 3),
                "max_confidence": round(student.max_conf, 3),
                "raw_track_ids": student.raw_ids,
                "reidentifications": student.rebinds,
                "home_zone": student.home_zone,
                "zone_time_s": zone_time,
                "zone_changes": max(0, len(student.zone_visits) - 1),
                "distance_px": round(student.distance_px, 1),
                "moving_s": round(student.moving_seconds, 2),
                "moving_ratio": round(
                    student.moving_seconds / student.duration, 3
                ) if student.duration > 0.5 else 0.0,
                "snapshot": student.snapshot_path,
                "trajectory_points": len(student.trajectory),
            })
        return rows

    def summary(self, registry: StudentRegistry, source: dict) -> dict:
        students = self.student_rows(registry)

        durations = [s["duration_s"] for s in students] or [0.0]
        zone_totals: dict[str, float] = {}
        for student in registry.confirmed_students:
            for zone, seconds in student.zone_time.items():
                zone_totals[zone] = zone_totals.get(zone, 0.0) + seconds

        return {
            "source": source,
            "processed_frames": self.processed_frames,
            "students_detected": len(students),
            "tracks_discarded": registry.discarded_tracks,
            "gate_rejections": self.total_gate_rejections,
            "peak_present": self.peak_present,
            "peak_present_at_s": round(self.peak_time, 2),
            "avg_duration_s": round(float(np.mean(durations)), 2),
            "median_duration_s": round(float(np.median(durations)), 2),
            "total_events": len(registry.events),
            "zone_totals_s": {k: round(v, 2) for k, v in sorted(
                zone_totals.items(), key=lambda item: item[1], reverse=True)},
            "occupancy": self.occupancy_rows(),
            "students": students,
            "events": [
                {
                    "frame": event.frame,
                    "time_s": round(event.time, 2),
                    "student_id": event.student_id,
                    "type": event.type,
                    "detail": event.detail,
                }
                for event in registry.events
            ],
        }
