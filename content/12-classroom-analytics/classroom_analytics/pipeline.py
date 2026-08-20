"""The pipeline: one frame in, analytics out.

    Camera / video
          |
      frame capture            video_source.py   (Phase 4)
          |
      preprocessing            preprocess.py     (Phase 2)
          |
      person detection         detector.py       (Phase 8)
          |
      multi-object tracking    detector.py       (Phase 9, BoT-SORT + ReID)
          |
      quality gates            detector.py       (area / aspect / ROI / duplicates)
          |
      student registry         registry.py       (Kalman, coasting, re-ID)
          |
      zones + movement         zones.py, motion.py (Phase 3, Phase 5)
          |
      statistics               analytics.py
          |
      storage                  storage.py        (SQLite + CSV + JSON + images)
          |
      visualization            visualizer.py     (Phase 1)

Order matters in one place: every measurement is taken from the *clean* frame,
and only then is a copy annotated. Drawing first would feed boxes and labels
into the background subtractor and into the students' appearance histograms.
"""

from __future__ import annotations

import time

import cv2

from .analytics import SessionAnalytics
from .config import AppConfig
from .detector import GateStats, PersonDetector
from .motion import MotionAnalyzer
from .preprocess import Preprocessor
from .registry import StudentRegistry
from .storage import SessionStore
from .video_source import VideoSource
from .visualizer import Visualizer
from .zones import ZoneMap


class ClassroomPipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.registry: StudentRegistry | None = None
        self.analytics: SessionAnalytics | None = None
        self.store: SessionStore | None = None

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.cfg

        source = VideoSource(cfg.source).open()
        print(f"[source] {source.describe()}")

        detector = PersonDetector(cfg).load()
        print(f"[model ] {detector.describe()}")

        preprocessor = Preprocessor(cfg.preprocess)
        if preprocessor.active:
            print("[input ] preprocessing enabled "
                  f"(clahe={cfg.preprocess.clahe}, denoise={cfg.preprocess.denoise})")

        zone_map = ZoneMap(cfg.zones, source.width, source.height)
        if zone_map.zones:
            print(f"[zones ] {len(zone_map.zones)} zones: {', '.join(zone_map.names)}")
        if zone_map.roi is not None:
            print("[zones ] ROI active - detections outside it are ignored")

        motion = MotionAnalyzer(cfg.motion)
        self.registry = StudentRegistry(cfg, source.fps)
        self.analytics = SessionAnalytics(cfg, source.width, source.height, source.fps)
        visualizer = Visualizer(cfg.view)

        # The annotated video contains one frame per *processed* frame, so with
        # a stride it must be written at a lower rate or it plays as time-lapse.
        output_fps = source.fps / max(1, cfg.source.frame_stride)

        self.store = SessionStore(cfg, {
            "source": source.label,
            "fps": output_fps,
            "source_fps": source.fps,
            "width": source.width,
            "height": source.height,
            "tracker": detector.tracker_name,
        })
        print(f"[output] {self.store.dir}")

        totals = GateStats()
        frame_count = 0
        last_timestamp = None
        started = time.time()
        fps_marker = started
        fps_frames = 0
        live_fps = 0.0
        paused = False
        interrupted = False
        last_frame_idx = 0
        last_timestamp_seen = 0.0

        try:
            for frame in source.frames():
                frame_count += 1
                last_frame_idx = frame.index
                last_timestamp_seen = frame.timestamp

                # --- time step for the Kalman filter -------------------------
                if last_timestamp is None:
                    dt = 1.0 / source.fps
                else:
                    dt = frame.timestamp - last_timestamp
                    if not (0.0 < dt < 1.0):
                        # A stalled live stream or a rewound file: fall back to
                        # the nominal step instead of poisoning the filter.
                        dt = 1.0 / source.fps
                last_timestamp = frame.timestamp

                # --- measure on the clean frame ------------------------------
                detect_input = preprocessor.apply(frame.image)
                detections, gates = detector.track(
                    detect_input,
                    roi_test=zone_map.in_roi if zone_map.roi is not None else None,
                    established=self.registry.is_established,
                )
                totals.merge(gates)

                motion.update(frame.image)

                students = self.registry.update(
                    detections, frame.image, frame.index, frame.timestamp, dt,
                    zone_map, motion,
                )

                self.analytics.update(frame.image, frame.timestamp, students,
                                      gates.rejected)
                self.store.add_samples(frame.index, frame.timestamp, students)

                # --- annotate a copy -----------------------------------------
                canvas = frame.image.copy()
                zone_map.draw(canvas)
                visualizer.draw_students(canvas, students)

                fps_frames += 1
                now = time.time()
                if now - fps_marker >= 0.5:
                    live_fps = fps_frames / (now - fps_marker)
                    fps_marker = now
                    fps_frames = 0

                visualizer.draw_hud(canvas, {
                    "title": "Classroom Analytics",
                    "present": self.registry.present_count(),
                    "total": self.registry.total_confirmed,
                    "moving": sum(1 for s in students if s.moving),
                    "predicted": sum(1 for s in students if s.state == "coasting"),
                    "frame": frame.index,
                    "time": frame.timestamp,
                    "fps": live_fps,
                    "rejected": totals.rejected,
                }, self.analytics.recent_occupancy)

                self.store.write_video_frame(canvas)

                # --- preview --------------------------------------------------
                if cfg.view.show:
                    visualizer.draw_hints(canvas, paused)
                    cv2.imshow(cfg.view.window, canvas)

                    key = cv2.waitKey(1) & 0xFF
                    while paused and key not in (ord(" "), ord("q")):
                        key = cv2.waitKey(30) & 0xFF

                    if key == ord("q"):
                        print("[run   ] stopped by user")
                        interrupted = True
                        break
                    if key == ord(" "):
                        paused = not paused
                    elif key == ord("h"):
                        visualizer.show_overlay = not visualizer.show_overlay

                # --- progress -------------------------------------------------
                every = max(1, cfg.view.progress_every)
                if frame_count % every == 0:
                    self._progress(frame_count, frame, source, live_fps, totals)

        except KeyboardInterrupt:
            print("\n[run   ] interrupted - writing what we have")
            interrupted = True
        finally:
            source.release()
            if cfg.view.show:
                cv2.destroyAllWindows()

        # --------------------------------------------------------- finalize
        self.registry.finalize(last_frame_idx, last_timestamp_seen)

        summary = self.analytics.summary(self.registry, {
            "source": source.label,
            "kind": "live" if source.is_live else "file",
            "fps": round(source.fps, 3),
            "resolution": [source.width, source.height],
            "total_frames": source.total_frames,
            "model": cfg.detect.model,
            "imgsz": cfg.detect.imgsz,
            "conf": cfg.detect.conf,
            "tracker": detector.tracker_name,
            "frame_stride": cfg.source.frame_stride,
            "zones": zone_map.to_dict(),
            "interrupted": interrupted,
        })
        summary["wall_clock_s"] = round(time.time() - started, 2)
        summary["processing_fps"] = round(
            frame_count / max(1e-6, time.time() - started), 2)
        summary["gate_breakdown"] = {
            "raw_detections": totals.raw,
            "kept": totals.kept,
            "no_track_id": totals.no_track_id,
            "too_small": totals.too_small,
            "bad_aspect": totals.bad_aspect,
            "outside_roi": totals.outside_roi,
            "duplicate": totals.duplicate,
        }

        artifacts = self.store.finalize(self.registry, self.analytics, summary)
        self._report(summary, artifacts)
        return {"summary": summary, "artifacts": artifacts}

    # -------------------------------------------------------------- printing
    def _progress(self, frame_count, frame, source, live_fps, totals: GateStats) -> None:
        assert self.registry is not None
        if source.total_frames:
            done = f"{frame.index}/{source.total_frames}"
            percent = 100.0 * frame.index / source.total_frames
            position = f"{done} ({percent:4.1f}%)"
        else:
            position = f"frame {frame.index} | t {frame.timestamp:6.1f}s"

        print(f"[run   ] {position} | present {self.registry.present_count():2d} "
              f"| total {self.registry.total_confirmed:2d} "
              f"| {live_fps:5.1f} fps | rejected {totals.rejected}")

    def _report(self, summary: dict, artifacts: dict) -> None:
        print("\n" + "=" * 62)
        print("SESSION SUMMARY")
        print("=" * 62)
        print(f"  session          : {artifacts['session_id']}")
        print(f"  source           : {summary['source']['source']}")
        print(f"  frames processed : {summary['processed_frames']} "
              f"in {summary['wall_clock_s']}s ({summary['processing_fps']} fps)")
        print(f"  students         : {summary['students_detected']} confirmed, "
              f"{summary['tracks_discarded']} noise tracks discarded")
        print(f"  peak in room     : {summary['peak_present']} "
              f"at t={summary['peak_present_at_s']}s")
        print(f"  avg time in room : {summary['avg_duration_s']}s "
              f"(median {summary['median_duration_s']}s)")
        print(f"  events logged    : {summary['total_events']}")

        gates = summary["gate_breakdown"]
        print(f"  quality gates    : kept {gates['kept']} of {gates['raw_detections']} "
              f"raw boxes (small {gates['too_small']}, aspect {gates['bad_aspect']}, "
              f"roi {gates['outside_roi']}, duplicate {gates['duplicate']}, "
              f"no-id {gates['no_track_id']})")

        if summary["zone_totals_s"]:
            top = list(summary["zone_totals_s"].items())[:4]
            print("  busiest zones    : "
                  + ", ".join(f"{name} {seconds:.0f}s" for name, seconds in top))

        print("-" * 62)
        for key in ("video", "csv", "occupancy", "json", "heatmap",
                    "snapshots", "database"):
            if key in artifacts:
                print(f"  {key:<9}: {artifacts[key]}")
        print("=" * 62)
