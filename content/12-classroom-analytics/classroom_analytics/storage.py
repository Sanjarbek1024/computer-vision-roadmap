"""Persistence: everything the session produces for later analysis.

Layout under `outputs/`:

    analytics.db                      one SQLite database, all sessions
    session_<stamp>/annotated.mp4     the rendered video
    session_<stamp>/tracks.csv        per-frame rows (spreadsheet friendly)
    session_<stamp>/occupancy.csv     occupancy timeline
    session_<stamp>/session.json      full summary incl. events + trajectories
    session_<stamp>/heatmap.jpg       where people spent their time
    session_<stamp>/students/student_07/snapshot.jpg

The SQLite database is the part that makes this "future analytics" rather than
"one video with boxes on it": every session appends to the same tables, so
questions like "average attendance in room 204 this month" are one query away.
CSV and JSON are written alongside it because a colleague with Excel should not
need a database client.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import AppConfig
from .registry import Student, StudentRegistry

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    source            TEXT,
    started_at        TEXT,
    ended_at          TEXT,
    fps               REAL,
    width             INTEGER,
    height            INTEGER,
    model             TEXT,
    imgsz             INTEGER,
    conf              REAL,
    tracker           TEXT,
    processed_frames  INTEGER,
    students_detected INTEGER,
    peak_present      INTEGER,
    tracks_discarded  INTEGER
);

CREATE TABLE IF NOT EXISTS students (
    session_id        TEXT,
    student_id        INTEGER,
    first_frame       INTEGER,
    last_frame        INTEGER,
    first_seen_s      REAL,
    last_seen_s       REAL,
    duration_s        REAL,
    visible_frames    INTEGER,
    predicted_frames  INTEGER,
    tracking_quality  REAL,
    avg_confidence    REAL,
    max_confidence    REAL,
    reidentifications INTEGER,
    raw_track_ids     TEXT,
    home_zone         TEXT,
    zone_changes      INTEGER,
    distance_px       REAL,
    moving_s          REAL,
    snapshot          TEXT,
    PRIMARY KEY (session_id, student_id)
);

CREATE TABLE IF NOT EXISTS track_samples (
    session_id  TEXT,
    frame       INTEGER,
    t_seconds   REAL,
    student_id  INTEGER,
    state       TEXT,
    x1          INTEGER,
    y1          INTEGER,
    x2          INTEGER,
    y2          INTEGER,
    cx          INTEGER,
    cy          INTEGER,
    vx          REAL,
    vy          REAL,
    speed       REAL,
    conf        REAL,
    zone        TEXT,
    moving      INTEGER,
    activity    REAL
);

CREATE TABLE IF NOT EXISTS events (
    session_id  TEXT,
    frame       INTEGER,
    t_seconds   REAL,
    student_id  INTEGER,
    type        TEXT,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS zone_visits (
    session_id  TEXT,
    student_id  INTEGER,
    zone        TEXT,
    start_s     REAL,
    end_s       REAL,
    duration_s  REAL
);

CREATE TABLE IF NOT EXISTS occupancy (
    session_id     TEXT,
    bucket_start_s REAL,
    bucket_end_s   REAL,
    samples        INTEGER,
    avg_present    REAL,
    peak_present   INTEGER,
    avg_moving     REAL
);

CREATE INDEX IF NOT EXISTS idx_samples_session ON track_samples(session_id, frame);
CREATE INDEX IF NOT EXISTS idx_samples_student ON track_samples(session_id, student_id);
CREATE INDEX IF NOT EXISTS idx_events_session  ON events(session_id, t_seconds);
"""

CSV_HEADER = [
    "frame", "time_s", "student_id", "state",
    "x1", "y1", "x2", "y2", "cx", "cy",
    "vx", "vy", "speed_px_s", "conf", "zone", "moving", "activity",
]


class SessionStore:
    """Writes one session to disk; appends to the shared SQLite database."""

    def __init__(self, cfg: AppConfig, meta: dict):
        self.cfg = cfg
        self.out = cfg.output
        self.meta = meta

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_id = self.out.session_name or f"session_{stamp}"
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        self.root = cfg.output_dir
        self.dir = self.root / self.session_id
        self.students_dir = self.dir / "students"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.dir / "annotated.mp4"
        self.csv_path = self.dir / "tracks.csv"
        self.json_path = self.dir / "session.json"
        self.occupancy_path = self.dir / "occupancy.csv"
        self.heatmap_path = self.dir / "heatmap.jpg"
        self.db_path = self.root / "analytics.db"

        self._writer: cv2.VideoWriter | None = None
        self._csv_file = None
        self._csv = None
        self._db: sqlite3.Connection | None = None
        self._pending: list[tuple] = []
        self._rows_written = 0

        self._open()

    # ------------------------------------------------------------------ open
    def _open(self) -> None:
        if self.out.save_csv:
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(CSV_HEADER)

        if self.out.save_sqlite:
            self._db = sqlite3.connect(self.db_path)
            self._db.executescript(SCHEMA)
            self._db.execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_id, source, started_at, fps, width, height, model, "
                " imgsz, conf, tracker, processed_frames, students_detected, "
                " peak_present, tracks_discarded) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,0,0,0,0)",
                (
                    self.session_id,
                    self.meta.get("source", ""),
                    self.started_at,
                    self.meta.get("source_fps", self.meta.get("fps", 0.0)),
                    self.meta.get("width", 0),
                    self.meta.get("height", 0),
                    self.cfg.detect.model,
                    self.cfg.detect.imgsz,
                    self.cfg.detect.conf,
                    self.meta.get("tracker", ""),
                ),
            )
            self._db.commit()

    # ----------------------------------------------------------------- video
    def write_video_frame(self, frame: np.ndarray) -> None:
        if not self.out.save_video:
            return

        if self._writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.out.video_fourcc)
            fps = float(self.meta.get("fps", 25.0)) or 25.0
            self._writer = cv2.VideoWriter(
                str(self.video_path), fourcc, fps, (width, height)
            )
            if not self._writer.isOpened():
                print(f"[storage] could not open video writer at {self.video_path}; "
                      f"continuing without video output")
                self.out.save_video = False
                self._writer = None
                return

        self._writer.write(frame)

    # --------------------------------------------------------------- samples
    def add_samples(self, frame_idx: int, timestamp: float,
                    students: list[Student]) -> None:
        """Record one row per visible student for this frame."""
        if not students:
            return

        every = max(1, self.out.sample_every)
        if frame_idx % every != 0:
            return

        for student in students:
            x1, y1, x2, y2 = (int(v) for v in student.box)
            cx, cy = (int(v) for v in student.center)
            vx, vy = student.kalman.velocity

            row = (
                frame_idx, round(timestamp, 3), student.student_id, student.state,
                x1, y1, x2, y2, cx, cy,
                round(vx, 2), round(vy, 2), round(student.kalman.speed, 2),
                round(student.conf, 3), student.zone,
                1 if student.moving else 0, round(student.activity, 4),
            )

            if self._csv is not None:
                self._csv.writerow(row)

            if self._db is not None:
                self._pending.append((self.session_id, *row))

            self._rows_written += 1

        if self._db is not None and len(self._pending) >= self.out.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._db is not None and self._pending:
            self._db.executemany(
                "INSERT INTO track_samples VALUES (" + ",".join(["?"] * 18) + ")",
                self._pending,
            )
            self._db.commit()
            self._pending.clear()
        if self._csv_file is not None:
            self._csv_file.flush()

    # -------------------------------------------------------------- finalize
    def finalize(self, registry: StudentRegistry, analytics, summary: dict) -> dict:
        """Write everything that is only known once the session has ended."""
        self.flush()

        if self.out.save_snapshots:
            self._write_snapshots(registry)
            # Snapshot paths are only known now, so refresh the summary rows.
            summary["students"] = analytics.student_rows(registry)

        if self.out.save_json:
            payload = dict(summary)
            payload["session_id"] = self.session_id
            payload["started_at"] = self.started_at
            payload["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            payload["trajectories"] = {
                str(student.student_id): [
                    {"frame": f, "t": round(t, 2), "x": x, "y": y}
                    for f, t, x, y in student.trajectory
                ]
                for student in registry.confirmed_students
            }
            with open(self.json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

        if self.out.save_csv:
            with open(self.occupancy_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["bucket_start_s", "bucket_end_s", "samples",
                                 "avg_present", "peak_present", "avg_moving"])
                for row in summary["occupancy"]:
                    writer.writerow([row["bucket_start_s"], row["bucket_end_s"],
                                     row["samples"], row["avg_present"],
                                     row["peak_present"], row["avg_moving"]])

        if self.out.save_heatmap:
            heatmap = analytics.heatmap_image()
            if heatmap is not None:
                cv2.imwrite(str(self.heatmap_path), heatmap)

        if self._db is not None:
            self._write_db(registry, summary)

        self.close()
        return self._artifacts()

    def _write_snapshots(self, registry: StudentRegistry) -> None:
        for student in registry.confirmed_students:
            if student.snapshot is None:
                continue
            folder = self.students_dir / f"student_{student.student_id:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "snapshot.jpg"
            cv2.imwrite(str(path), student.snapshot)
            student.snapshot_path = str(path.relative_to(self.root))

    def _write_db(self, registry: StudentRegistry, summary: dict) -> None:
        assert self._db is not None
        db = self._db

        db.execute(
            "UPDATE sessions SET ended_at=?, processed_frames=?, "
            "students_detected=?, peak_present=?, tracks_discarded=? "
            "WHERE session_id=?",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                summary["processed_frames"],
                summary["students_detected"],
                summary["peak_present"],
                summary["tracks_discarded"],
                self.session_id,
            ),
        )

        db.execute("DELETE FROM students WHERE session_id=?", (self.session_id,))
        db.executemany(
            "INSERT INTO students VALUES (" + ",".join(["?"] * 19) + ")",
            [
                (
                    self.session_id, row["student_id"], row["first_frame"],
                    row["last_frame"], row["first_seen_s"], row["last_seen_s"],
                    row["duration_s"], row["visible_frames"], row["predicted_frames"],
                    row["tracking_quality"], row["avg_confidence"],
                    row["max_confidence"], row["reidentifications"],
                    ",".join(str(i) for i in row["raw_track_ids"]),
                    row["home_zone"], row["zone_changes"], row["distance_px"],
                    row["moving_s"], row["snapshot"],
                )
                for row in summary["students"]
            ],
        )

        db.execute("DELETE FROM events WHERE session_id=?", (self.session_id,))
        db.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            [
                (self.session_id, event.frame, round(event.time, 3),
                 event.student_id, event.type, event.detail)
                for event in registry.events
            ],
        )

        db.execute("DELETE FROM zone_visits WHERE session_id=?", (self.session_id,))
        db.executemany(
            "INSERT INTO zone_visits VALUES (?,?,?,?,?,?)",
            [
                (self.session_id, student.student_id, visit.zone,
                 round(visit.start_time, 3), round(visit.end_time, 3),
                 round(visit.duration, 3))
                for student in registry.confirmed_students
                for visit in student.zone_visits
                if visit.end_time > 0
            ],
        )

        db.execute("DELETE FROM occupancy WHERE session_id=?", (self.session_id,))
        db.executemany(
            "INSERT INTO occupancy VALUES (?,?,?,?,?,?,?)",
            [
                (self.session_id, row["bucket_start_s"], row["bucket_end_s"],
                 row["samples"], row["avg_present"], row["peak_present"],
                 row["avg_moving"])
                for row in summary["occupancy"]
            ],
        )

        db.commit()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv = None
        if self._db is not None:
            self._db.close()
            self._db = None

    def _artifacts(self) -> dict:
        items = {"session_id": self.session_id, "folder": self.dir}
        if self.out.save_video:
            items["video"] = self.video_path
        if self.out.save_csv:
            items["csv"] = self.csv_path
            items["occupancy"] = self.occupancy_path
        if self.out.save_json:
            items["json"] = self.json_path
        if self.out.save_sqlite:
            items["database"] = self.db_path
        if self.out.save_heatmap and self.heatmap_path.exists():
            items["heatmap"] = self.heatmap_path
        if self.out.save_snapshots and self.students_dir.exists():
            items["snapshots"] = self.students_dir
        items["rows"] = self._rows_written
        return items
