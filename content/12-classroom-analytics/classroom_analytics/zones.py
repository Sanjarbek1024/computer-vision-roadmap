"""Classroom zones and region of interest (Phase 3: contours, ROI, polygons).

Zones turn raw pixel coordinates into something a teacher can read: not
"student 7 at (1180, 640)" but "student 7 in back-right". They are defined in
*normalized* coordinates (0..1), so the same config file works whether the
camera streams 1080p or 4K.

Two things are configurable:

    roi       - one polygon. Anything whose feet land outside it is ignored
                completely (e.g. a corridor visible through an open door).
    polygons  - named zones. If none are given, a rows x cols grid is generated.

Membership is judged on the *bottom-centre* of the box - where the student
actually stands - not the box centre, which floats around chest height.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ZoneConfig


@dataclass
class Zone:
    name: str
    polygon: np.ndarray          # pixel coordinates, int32, shape (N, 2)

    @property
    def centroid(self) -> tuple[int, int]:
        point = self.polygon.mean(axis=0)
        return int(point[0]), int(point[1])

    def contains(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0


def _to_pixels(polygon: list[list[float]], width: int, height: int) -> np.ndarray:
    points = [(float(x) * width, float(y) * height) for x, y in polygon]
    return np.array(points, dtype=np.int32)


class ZoneMap:
    """Named regions of the classroom plus an optional ROI."""

    def __init__(self, cfg: ZoneConfig, width: int, height: int):
        self.cfg = cfg
        self.width = width
        self.height = height

        self.zones: list[Zone] = []
        self.roi: Zone | None = None

        if cfg.roi:
            self.roi = Zone("roi", _to_pixels(cfg.roi, width, height))

        if not cfg.enabled:
            return

        if cfg.polygons:
            for name, polygon in cfg.polygons.items():
                if len(polygon) >= 3:
                    self.zones.append(Zone(name, _to_pixels(polygon, width, height)))
        else:
            self.zones = self._build_grid(cfg, width, height)

    # ------------------------------------------------------------------ grid
    @staticmethod
    def _build_grid(cfg: ZoneConfig, width: int, height: int) -> list[Zone]:
        zones: list[Zone] = []
        rows = max(1, cfg.grid_rows)
        cols = max(1, cfg.grid_cols)

        for r in range(rows):
            row_name = cfg.row_names[r] if r < len(cfg.row_names) else f"row{r + 1}"
            for c in range(cols):
                col_name = cfg.col_names[c] if c < len(cfg.col_names) else f"col{c + 1}"

                x1 = int(width * c / cols)
                x2 = int(width * (c + 1) / cols)
                y1 = int(height * r / rows)
                y2 = int(height * (r + 1) / rows)

                polygon = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                )
                zones.append(Zone(f"{row_name}-{col_name}", polygon))

        return zones

    # ----------------------------------------------------------------- query
    def zone_for(self, point: tuple[float, float]) -> str:
        for zone in self.zones:
            if zone.contains(point):
                return zone.name
        return "unassigned"

    def in_roi(self, point: tuple[float, float]) -> bool:
        return True if self.roi is None else self.roi.contains(point)

    @property
    def names(self) -> list[str]:
        return [zone.name for zone in self.zones]

    # --------------------------------------------------------------- drawing
    def draw(self, frame: np.ndarray) -> None:
        """Thin outlines + small labels. Deliberately quiet: the students are
        the subject of the frame, the zones are just reference lines."""
        if not self.cfg.draw:
            return

        for zone in self.zones:
            cv2.polylines(frame, [zone.polygon], True, (70, 70, 70), 1, cv2.LINE_AA)
            cx, cy = zone.centroid
            cv2.putText(frame, zone.name, (cx - 34, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)

        if self.roi is not None:
            cv2.polylines(frame, [self.roi.polygon], True, (0, 140, 255), 2, cv2.LINE_AA)

    def to_dict(self) -> dict:
        return {
            "roi": self.roi.polygon.tolist() if self.roi else None,
            "zones": {zone.name: zone.polygon.tolist() for zone in self.zones},
        }
