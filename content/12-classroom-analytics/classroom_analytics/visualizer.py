"""Drawing and the on-screen dashboard (Phase 1: shapes, text, colour).

Visual language, chosen so a glance tells you how much to trust what you see:

    solid box    - the student was detected in this frame
    dashed box   - the student is occluded; the box is a Kalman prediction
    trail        - recent smoothed path, fading out with age
    filled dot   - the student is moving (walking, or high activity in place)

The HUD carries the numbers a demo needs to be believable: live FPS, how many
students are present now, how many have been seen in total, and how many raw
detections the quality gates rejected.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import ViewConfig
from .registry import Student

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
DARK = (28, 28, 28)


def draw_dashed_rect(frame, pt1, pt2, color, thickness=1, dash=9, gap=6) -> None:
    """A dashed rectangle - our marker for 'this box is predicted, not measured'."""
    x1, y1 = pt1
    x2, y2 = pt2

    def line(start, end, horizontal: bool) -> None:
        length = end[0] - start[0] if horizontal else end[1] - start[1]
        step = dash + gap
        for offset in range(0, max(1, abs(int(length))), step):
            a = (start[0] + offset, start[1]) if horizontal else (start[0], start[1] + offset)
            stop = min(offset + dash, abs(int(length)))
            b = (start[0] + stop, start[1]) if horizontal else (start[0], start[1] + stop)
            cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)

    line((x1, y1), (x2, y1), True)
    line((x1, y2), (x2, y2), True)
    line((x1, y1), (x1, y2), False)
    line((x2, y1), (x2, y2), False)


def draw_label(frame, origin, text, color, font_scale=0.45, thickness=1) -> None:
    """Filled chip with text, flipped below the anchor if it would leave the frame."""
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    top = y - th - baseline - 4
    if top < 0:
        top = y + 4

    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 4), color, -1)
    cv2.putText(frame, text, (x + 4, top + th + 2), FONT, font_scale,
                WHITE, thickness, cv2.LINE_AA)


class Visualizer:
    def __init__(self, cfg: ViewConfig):
        self.cfg = cfg
        self.show_overlay = True

    # --------------------------------------------------------------- students
    def draw_students(self, frame: np.ndarray, students: list[Student]) -> None:
        if not self.show_overlay:
            return

        for student in students:
            x1, y1, x2, y2 = (int(v) for v in student.box)
            color = student.color
            predicted = student.state == "coasting"

            if predicted:
                if not self.cfg.draw_predicted:
                    continue
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color,
                                 max(1, self.cfg.box_thickness - 1))
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                              self.cfg.box_thickness, cv2.LINE_AA)

            label = student.label
            if predicted:
                label += " ?"
            elif student.conf:
                label += f" {student.conf:.2f}"

            draw_label(frame, (x1, y1), label, color, self.cfg.font_scale)

            # Movement marker at the student's feet.
            ax, ay = (int(v) for v in student.anchor)
            if student.moving:
                cv2.circle(frame, (ax, ay), 4, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (ax, ay), 3, color, 1, cv2.LINE_AA)

            if self.cfg.draw_trail:
                self._draw_trail(frame, student)

    def _draw_trail(self, frame: np.ndarray, student: Student) -> None:
        points = student.trajectory[-self.cfg.trail_length:]
        if len(points) < 2:
            return

        color = student.color
        total = len(points)
        for index in range(1, total):
            _, _, x0, y0 = points[index - 1]
            _, _, x1, y1 = points[index]
            fade = index / total          # older segments are dimmer and thinner
            shade = tuple(int(c * (0.25 + 0.75 * fade)) for c in color)
            cv2.line(frame, (x0, y0), (x1, y1), shade,
                     1 if fade < 0.6 else 2, cv2.LINE_AA)

    # -------------------------------------------------------------------- HUD
    def draw_hud(self, frame: np.ndarray, stats: dict,
                 occupancy: list[int] | None = None) -> None:
        if not (self.cfg.draw_hud and self.show_overlay):
            return

        lines = [
            f"present {stats.get('present', 0)}   total {stats.get('total', 0)}",
            f"moving {stats.get('moving', 0)}   predicted {stats.get('predicted', 0)}",
            f"frame {stats.get('frame', 0)}   t {stats.get('time', 0.0):.1f}s",
            f"{stats.get('fps', 0.0):.1f} fps   rejected {stats.get('rejected', 0)}",
        ]

        pad = 10
        width = 250
        height = pad * 2 + 22 + len(lines) * 18 + (26 if occupancy else 0)
        left, top = self._hud_origin(frame, width, height, pad)

        panel = frame[top:top + height, left:left + width]
        if panel.shape[0] == height and panel.shape[1] == width:
            # Nearly opaque on purpose: fixed cameras burn a timestamp into a
            # corner, and a translucent card lets it bleed through the text.
            opacity = min(1.0, max(0.0, self.cfg.hud_opacity))
            overlay = np.full_like(panel, DARK, dtype=np.uint8)
            cv2.addWeighted(overlay, opacity, panel, 1.0 - opacity, 0, panel)

        cv2.rectangle(frame, (left, top), (left + width, top + height),
                      (60, 60, 60), 1)

        y = top + 22
        cv2.putText(frame, stats.get("title", "Classroom Analytics"),
                    (left + 10, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        y += 8

        for line in lines:
            y += 18
            cv2.putText(frame, line, (left + 10, y), FONT, 0.42, GREY, 1, cv2.LINE_AA)

        if occupancy:
            self._draw_sparkline(frame, occupancy,
                                 (left + 10, y + 8), width - 20, 16)

    def _hud_origin(self, frame: np.ndarray, width: int, height: int,
                    pad: int) -> tuple[int, int]:
        """Top-left corner of the HUD card for the configured position."""
        frame_h, frame_w = frame.shape[:2]
        position = (self.cfg.hud_position or "top-left").lower()

        left = pad
        top = pad
        if "right" in position:
            left = max(pad, frame_w - width - pad)
        if "bottom" in position:
            # Leave room for the key hints printed along the bottom edge.
            top = max(pad, frame_h - height - pad - 24)

        return left, top

    @staticmethod
    def _draw_sparkline(frame, values: list[int], origin, width: int, height: int) -> None:
        """Occupancy over the last few hundred frames, as a tiny bar chart."""
        if not values:
            return

        x0, y0 = origin
        peak = max(1, max(values))
        count = min(len(values), width)
        sliced = values[-count:]

        for index, value in enumerate(sliced):
            bar = int((value / peak) * height)
            x = x0 + index
            cv2.line(frame, (x, y0 + height), (x, y0 + height - bar),
                     (90, 170, 255), 1)

        cv2.putText(frame, f"peak {peak}", (x0 + width - 52, y0 + height),
                    FONT, 0.35, GREY, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ hints
    @staticmethod
    def draw_hints(frame: np.ndarray, paused: bool) -> None:
        text = "[q] quit   [space] pause   [h] overlay" + ("   PAUSED" if paused else "")
        height = frame.shape[0]
        cv2.putText(frame, text, (12, height - 12), FONT, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
