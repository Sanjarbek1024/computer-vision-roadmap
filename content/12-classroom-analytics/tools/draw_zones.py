#!/usr/bin/env python
"""Draw classroom zones and the ROI by clicking on a frame.

Zones in app.yaml are normalized polygons, which nobody wants to type by hand.
This grabs one frame from the source and lets you click the corners; it prints
YAML you can paste straight into configs/app.yaml.

    python tools/draw_zones.py                             # grab from the camera
    python tools/draw_zones.py --source ../11-classroom-monitoring/demo/class_videos/camera_test.mp4

Controls:

    left click   add a point
    right click  undo the last point
    n            finish this polygon and name it in the terminal
    r            reset the current polygon
    s            save the YAML snippet next to the config
    q            quit

Uses the mouse-callback and polygon drawing from Phase 1/Phase 3 of the roadmap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classroom_analytics.config import PROJECT_DIR, SourceConfig, load_dotenv  # noqa: E402
from classroom_analytics.video_source import VideoSource  # noqa: E402

WINDOW = "draw zones - [n]ame polygon  [r]eset  [s]ave  [q]uit"


class PolygonEditor:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.height, self.width = frame.shape[:2]
        self.current: list[tuple[int, int]] = []
        self.polygons: dict[str, list[tuple[int, int]]] = {}

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.current:
            self.current.pop()

    def render(self) -> np.ndarray:
        canvas = self.frame.copy()

        for name, points in self.polygons.items():
            array = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [array], True, (0, 200, 255), 2, cv2.LINE_AA)
            cx, cy = array.mean(axis=0).astype(int)
            cv2.putText(canvas, name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2, cv2.LINE_AA)

        for index, point in enumerate(self.current):
            cv2.circle(canvas, point, 4, (0, 255, 0), -1, cv2.LINE_AA)
            if index:
                cv2.line(canvas, self.current[index - 1], point,
                         (0, 255, 0), 2, cv2.LINE_AA)
        if len(self.current) > 2:
            cv2.line(canvas, self.current[-1], self.current[0],
                     (0, 160, 0), 1, cv2.LINE_AA)

        hint = f"{len(self.polygons)} polygon(s) | {len(self.current)} point(s)"
        cv2.putText(canvas, hint, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def to_yaml(self) -> str:
        """Normalized coordinates, so the config survives a resolution change."""
        lines = ["zones:", "  polygons:"]
        roi: list[tuple[int, int]] | None = None

        for name, points in self.polygons.items():
            if name == "roi":
                roi = points
                continue
            coords = ", ".join(
                f"[{x / self.width:.4f}, {y / self.height:.4f}]" for x, y in points
            )
            lines.append(f"    {name}: [{coords}]")

        if len(lines) == 2:
            lines.append("    {}")

        if roi is not None:
            coords = ", ".join(
                f"[{x / self.width:.4f}, {y / self.height:.4f}]" for x, y in roi
            )
            lines.append(f"  roi: [{coords}]")

        return "\n".join(lines) + "\n"


def grab_frame(source_path: str, frame_number: int) -> np.ndarray:
    cfg = SourceConfig(path=source_path, fallback="", start_frame=frame_number,
                       drop_late_frames=False, reconnect_attempts=2)
    source = VideoSource(cfg).open()
    try:
        for frame in source.frames():
            return frame.image
    finally:
        source.release()
    raise SystemExit("Could not read a frame from the source.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive zone/ROI editor.")
    parser.add_argument("--source", default="env:CLASSROOM_RTSP_URL")
    parser.add_argument("--frame", type=int, default=0, help="which frame to grab")
    parser.add_argument("--out", default="configs/zones_snippet.yaml")
    args = parser.parse_args(argv)

    load_dotenv()

    frame = grab_frame(args.source, args.frame)
    editor = PolygonEditor(frame)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1600, editor.width), min(900, editor.height))
    cv2.setMouseCallback(WINDOW, editor.on_mouse)

    print(__doc__)
    print(f"frame size: {editor.width}x{editor.height}")
    print("Tip: name a polygon 'roi' to make it the region of interest.\n")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_DIR / out_path

    while True:
        cv2.imshow(WINDOW, editor.render())
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            editor.current.clear()
        elif key == ord("n"):
            if len(editor.current) < 3:
                print("need at least 3 points")
                continue
            name = input("polygon name (e.g. front_row, teacher_desk, roi): ").strip()
            if name:
                editor.polygons[name] = list(editor.current)
                editor.current.clear()
                print(f"  added '{name}'")
        elif key == ord("s"):
            if not editor.polygons:
                print("nothing to save yet")
                continue
            snippet = editor.to_yaml()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(snippet, encoding="utf-8")
            print(f"\nsaved -> {out_path}\n")
            print(snippet)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
