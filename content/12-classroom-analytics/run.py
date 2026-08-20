#!/usr/bin/env python
"""Entry point for the classroom analytics demo.

    python run.py                                   # config defaults
    python run.py --source class_videos/lesson.mp4  # a recording
    python run.py --source 0                        # webcam
    python run.py --source env:CLASSROOM_RTSP_URL   # the wall camera
    python run.py --preset realtime --no-video      # live monitoring

Everything is configurable in configs/app.yaml; the flags below are the ones
worth changing from the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow "python run.py" from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classroom_analytics import ClassroomPipeline, load_config, load_dotenv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classroom analytics: detect, track and log students.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_argument_group("source")
    source.add_argument("--source", help="video file, rtsp:// URL, webcam index, "
                                         "or env:VAR_NAME")
    source.add_argument("--stride", type=int, help="process every Nth frame")
    source.add_argument("--start-frame", type=int, help="skip the first N frames")
    source.add_argument("--max-frames", type=int, help="stop after N processed frames")

    model = parser.add_argument_group("detection / tracking")
    model.add_argument("--model", help="YOLO weights (default: yolo11n.pt)")
    model.add_argument("--imgsz", type=int, help="inference size; higher finds "
                                                 "smaller students")
    model.add_argument("--conf", type=float, help="detector confidence floor")
    model.add_argument("--tracker", help="tracker YAML, relative to the project")
    model.add_argument("--min-hits", type=int,
                       help="frames a track must survive before it becomes a student")
    model.add_argument("--confirm-conf", type=float,
                       help="smoothed confidence needed to confirm a student")
    model.add_argument("--coast-frames", type=int,
                       help="frames a student is held on Kalman prediction alone")

    toggles = parser.add_argument_group("toggles")
    toggles.add_argument("--no-view", action="store_true", help="no preview window")
    toggles.add_argument("--no-video", action="store_true", help="do not write the "
                                                                "annotated video")
    toggles.add_argument("--no-db", action="store_true", help="skip SQLite")
    toggles.add_argument("--no-snapshots", action="store_true")
    toggles.add_argument("--no-zones", action="store_true")
    toggles.add_argument("--no-motion", action="store_true")
    toggles.add_argument("--no-reid", action="store_true", help="disable re-binding "
                                                               "of returning students")
    toggles.add_argument("--no-kalman-gate", action="store_true",
                         help="accept every detection, even implausible jumps")
    toggles.add_argument("--clahe", action="store_true",
                         help="local contrast boost for dim rooms")

    output = parser.add_argument_group("output")
    output.add_argument("--out", help="output directory")
    output.add_argument("--session", help="session name (default: timestamp)")
    output.add_argument("--config", default="configs/app.yaml",
                        help="YAML config; pass '' to use built-in defaults")
    output.add_argument("--preset", choices=["quality", "realtime"],
                        help="quality = imgsz 960; realtime = imgsz 640 + skip frames")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    load_dotenv()

    config_path = args.config or None
    if config_path and not (Path(config_path).is_absolute()
                            or (Path(__file__).parent / config_path).exists()):
        print(f"[config] {config_path} not found; using built-in defaults")
        config_path = None

    cfg, warnings = load_config(config_path)
    for warning in warnings:
        print(f"[config] {warning}")

    # ---- presets first, so explicit flags can still override them ----------
    if args.preset == "realtime":
        cfg.detect.imgsz = 640
        cfg.source.drop_late_frames = True
        cfg.output.sample_every = 2
    elif args.preset == "quality":
        cfg.detect.imgsz = max(cfg.detect.imgsz, 960)
        cfg.source.frame_stride = 1

    # ---- source ------------------------------------------------------------
    if args.source:
        cfg.source.path = args.source
    if args.stride:
        cfg.source.frame_stride = args.stride
    if args.start_frame is not None:
        cfg.source.start_frame = args.start_frame
    if args.max_frames is not None:
        cfg.source.max_frames = args.max_frames

    # ---- detection / tracking ---------------------------------------------
    if args.model:
        cfg.detect.model = args.model
    if args.imgsz:
        cfg.detect.imgsz = args.imgsz
    if args.conf is not None:
        cfg.detect.conf = args.conf
    if args.tracker:
        cfg.track.tracker = args.tracker
    if args.min_hits is not None:
        cfg.track.min_hits = args.min_hits
    if args.confirm_conf is not None:
        cfg.track.confirm_conf = args.confirm_conf
    if args.coast_frames is not None:
        cfg.track.max_coast_frames = args.coast_frames

    # ---- toggles -----------------------------------------------------------
    if args.no_view:
        cfg.view.show = False
    if args.no_video:
        cfg.output.save_video = False
    if args.no_db:
        cfg.output.save_sqlite = False
    if args.no_snapshots:
        cfg.output.save_snapshots = False
    if args.no_zones:
        cfg.zones.enabled = False
    if args.no_motion:
        cfg.motion.enabled = False
    if args.no_reid:
        cfg.reid.enabled = False
    if args.no_kalman_gate:
        cfg.kalman.reject_outliers = False
    if args.clahe:
        cfg.preprocess.clahe = True

    # ---- output ------------------------------------------------------------
    if args.out:
        cfg.output.dir = args.out
    if args.session:
        cfg.output.session_name = args.session

    try:
        ClassroomPipeline(cfg).run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\n[error ] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
