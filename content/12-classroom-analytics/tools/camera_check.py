#!/usr/bin/env python
"""Check the camera before running the full pipeline.

The original throwaway version of this script had the RTSP password typed into
line 3. This one reads it from the environment (or a .env file), so the URL can
live outside the repository:

    CLASSROOM_RTSP_URL=rtsp://user:password@192.168.1.35:554/Streaming/Channels/101

Usage:

    python tools/camera_check.py                 # env:CLASSROOM_RTSP_URL
    python tools/camera_check.py --source 0      # webcam
    python tools/camera_check.py --seconds 5 --no-view --save frame.jpg

What it reports: whether the stream opens, its resolution and FPS, how many
frames actually arrive per second (cameras lie about FPS), and how many reads
failed. If this looks wrong, nothing downstream will look right.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classroom_analytics.config import SourceConfig, load_dotenv  # noqa: E402
from classroom_analytics.video_source import VideoSource  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Camera / stream connectivity check.")
    parser.add_argument("--source", default="env:CLASSROOM_RTSP_URL",
                        help="rtsp:// URL, webcam index, file path, or env:VAR_NAME")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="how long to sample the stream")
    parser.add_argument("--no-view", action="store_true", help="no preview window")
    parser.add_argument("--save", help="write one frame to this path")
    args = parser.parse_args(argv)

    load_dotenv()

    cfg = SourceConfig(path=args.source, fallback="", reconnect_attempts=2)

    try:
        source = VideoSource(cfg).open()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"FAILED: {exc}")
        print("\nChecklist:")
        print("  - is CLASSROOM_RTSP_URL set (or .env present)?")
        print("  - can you ping the camera?")
        print("  - is the channel path right? Hikvision: "
              "/Streaming/Channels/101 (main) or /102 (sub)")
        return 1

    print(f"CONNECTED: {source.describe()}")

    started = time.time()
    frames = 0
    first = None

    try:
        for frame in source.frames():
            frames += 1
            if first is None:
                first = frame.image

            if not args.no_view:
                preview = frame.image
                if preview.shape[1] > 1280:
                    scale = 1280 / preview.shape[1]
                    preview = cv2.resize(preview, None, fx=scale, fy=scale)
                cv2.putText(preview, f"frame {frames}  press q to stop",
                            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("camera check", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if time.time() - started >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - started)
    print(f"  frames received : {frames} in {elapsed:.1f}s "
          f"-> {frames / elapsed:.2f} fps measured "
          f"(camera reports {source.fps:.2f})")

    if args.save and first is not None:
        cv2.imwrite(args.save, first)
        print(f"  saved frame     : {args.save}")

    if frames == 0:
        print("  NO FRAMES: the stream opened but delivered nothing. "
              "Try rtsp_transport=udp, or the sub-stream (channel 102).")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
