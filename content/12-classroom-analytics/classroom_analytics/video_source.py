"""Video input (Phase 4).

One class handles all three sources the demo cares about:

    file   -> deterministic, timestamps come from the frame index and FPS
    RTSP   -> live, reconnects on failure, always serves the newest frame
    webcam -> live, same handling as RTSP

The important live-camera detail: if the pipeline is slower than the camera,
`cap.read()` returns older and older frames and the preview drifts minutes
behind reality. The background reader below keeps only the most recent frame,
so the app always analyses "now" and simply skips what it could not keep up
with.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from .config import PROJECT_DIR, SourceConfig


@dataclass
class Frame:
    index: int          # index in the source (not the count of processed frames)
    timestamp: float    # seconds since the start of the session
    image: "cv2.typing.MatLike"


def resolve_source(cfg: SourceConfig) -> tuple[str | int, str]:
    """Turn the configured source into something cv2.VideoCapture understands.

    Returns (capture_argument, human_readable_label).
    """
    raw = (cfg.path or "").strip()

    # "env:VAR" keeps camera credentials out of the repository.
    if raw.lower().startswith("env:"):
        raw = os.environ.get(raw[4:].strip(), "").strip()

    if not raw:
        raw = (cfg.fallback or "").strip()

    if not raw:
        raise ValueError(
            "No video source configured. Set source.path in configs/app.yaml, "
            "pass --source, or define CLASSROOM_RTSP_URL in a .env file."
        )

    # Webcam index
    if raw.isdigit():
        return int(raw), f"webcam:{raw}"

    lowered = raw.lower()
    if lowered.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return raw, _mask_credentials(raw)

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_DIR / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    return str(path), str(path)


def _mask_credentials(url: str) -> str:
    """rtsp://admin:secret@host/... -> rtsp://***@host/... for safe logging."""
    if "@" not in url or "//" not in url:
        return url
    scheme, rest = url.split("//", 1)
    _, host = rest.split("@", 1)
    return f"{scheme}//***@{host}"


def is_live_source(target: str | int) -> bool:
    if isinstance(target, int):
        return True
    return target.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


class VideoSource:
    """Frame provider with reconnect + latest-frame semantics for live feeds."""

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg
        self.target, self.label = resolve_source(cfg)
        self.is_live = is_live_source(self.target)

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: tuple[int, "cv2.typing.MatLike"] | None = None
        self._stop = threading.Event()
        self._read_errors = 0

        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0
        self.started_at: float = 0.0

    # ------------------------------------------------------------------ open
    def open(self) -> "VideoSource":
        if self.is_live and self.cfg.rtsp_transport:
            # Must be set before the capture is created; FFMPEG reads it once.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;{self.cfg.rtsp_transport}",
            )

        self._cap = self._open_capture()

        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        if not (1.0 <= self.fps <= 240.0):
            self.fps = 25.0        # some RTSP cameras report 0 or nonsense
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        if not self.is_live and self.cfg.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.cfg.start_frame))

        self.started_at = time.time()

        if self.is_live and self.cfg.drop_late_frames:
            self._stop.clear()
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

        return self

    def _open_capture(self) -> cv2.VideoCapture:
        backend = cv2.CAP_FFMPEG if isinstance(self.target, str) and self.is_live else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.target, backend)

        if self.is_live:
            # A 1-frame buffer is the other half of the "always analyse now" fix.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.label}")
        return cap

    def _reconnect(self) -> bool:
        """Live sources only: rebuild the capture after a read failure."""
        for attempt in range(1, self.cfg.reconnect_attempts + 1):
            if self._stop.is_set():
                return False
            print(f"[source] reconnecting to {self.label} "
                  f"({attempt}/{self.cfg.reconnect_attempts})...")
            try:
                if self._cap is not None:
                    self._cap.release()
                self._cap = self._open_capture()
                print("[source] reconnected")
                return True
            except RuntimeError:
                time.sleep(self.cfg.reconnect_delay_s)
        print("[source] giving up on reconnect")
        return False

    # ------------------------------------------------------------ background
    def _reader_loop(self) -> None:
        index = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break

            ok, frame = cap.read()
            if not ok:
                self._read_errors += 1
                if not self._reconnect():
                    break
                continue

            index += 1
            with self._lock:
                self._latest = (index, frame)

    # ---------------------------------------------------------------- frames
    def frames(self) -> Iterator[Frame]:
        """Yield frames, already honouring frame_stride and max_frames."""
        stride = max(1, int(self.cfg.frame_stride))
        limit = int(self.cfg.max_frames)
        produced = 0
        last_served = -1

        while True:
            if limit and produced >= limit:
                return

            if self._thread is not None:
                # Live + threaded: take whatever the camera has right now.
                with self._lock:
                    latest = self._latest
                if latest is None:
                    if not self._thread.is_alive():
                        return
                    time.sleep(0.005)
                    continue
                index, image = latest
                if index == last_served:
                    if not self._thread.is_alive():
                        return
                    time.sleep(0.003)
                    continue
                last_served = index
                image = image.copy()
                timestamp = time.time() - self.started_at
            else:
                cap = self._cap
                if cap is None:
                    return

                ok, image = cap.read()
                if not ok:
                    if self.is_live and self._reconnect():
                        continue
                    return

                index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if stride > 1 and index % stride != 0:
                    continue

                timestamp = (
                    time.time() - self.started_at
                    if self.is_live
                    else index / self.fps
                )

            produced += 1
            yield Frame(index=index, timestamp=timestamp, image=image)

    # --------------------------------------------------------------- cleanup
    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.release()

    def describe(self) -> str:
        kind = "live" if self.is_live else "file"
        frames = f"{self.total_frames} frames" if self.total_frames else "unbounded"
        return (f"{self.label} [{kind}] {self.width}x{self.height} "
                f"@ {self.fps:.2f} fps, {frames}")
