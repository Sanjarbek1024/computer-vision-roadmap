"""Configuration for the classroom analytics app.

Every tunable lives here as a dataclass with a sane default, so the app runs
with zero configuration. `configs/app.yaml` overrides the defaults, and CLI
flags override the YAML. Nothing is read from a global at runtime - the whole
config object is passed down into the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


# Repo layout: <ROOT>/content/12-classroom-analytics/classroom_analytics/config.py
PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent.parent


@dataclass
class SourceConfig:
    """Where frames come from: a video file, an RTSP camera, or a webcam index."""

    # A path, an "rtsp://..." URL, a webcam index ("0"), or "env:VAR_NAME" to
    # read the value from the environment (keeps camera passwords out of git).
    path: str = "env:CLASSROOM_RTSP_URL"

    # Fallback used when `path` resolves to nothing - handy for offline demos.
    fallback: str = "../11-classroom-monitoring/demo/class_videos/camera_test.mp4"

    frame_stride: int = 1        # process every Nth frame (2 = half the work)
    start_frame: int = 0         # skip the first N frames of a file
    max_frames: int = 0          # 0 = run to the end of the source

    # Live streams only: read in a background thread and always keep the newest
    # frame. Without this, a slow pipeline makes an RTSP feed drift minutes behind.
    drop_late_frames: bool = True
    reconnect_attempts: int = 10
    reconnect_delay_s: float = 3.0

    # Force TCP for RTSP. UDP drops packets and produces smeared frames, which
    # the detector then reads as low-confidence garbage.
    rtsp_transport: str = "tcp"


@dataclass
class PreprocessConfig:
    """Frame enhancement before detection (Phase 2)."""

    clahe: bool = False          # local contrast - helps in dim/backlit rooms
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise: bool = False        # light median blur; removes sensor speckle
    denoise_ksize: int = 3


@dataclass
class DetectConfig:
    """YOLO person detection (Phase 8)."""

    model: str = "yolo11n.pt"    # nano: small and fast enough for live video
    imgsz: int = 960             # above the 640 default -> far better recall on
                                 # small, seated, background students
    conf: float = 0.10           # deliberately LOW: the tracker's second
                                 # association stage needs weak detections to
                                 # survive occlusion. Display filtering happens
                                 # later, per track, not per frame.
    iou: float = 0.7             # NMS overlap. Permissive, so two students
                                 # sitting shoulder-to-shoulder aren't merged.
    max_det: int = 100
    device: str = ""             # "" = auto, "cpu", "0" for the first GPU
    classes: list[int] = field(default_factory=lambda: [0])   # 0 = person

    # ---- geometry gates: reject boxes that cannot be a student -------------
    # Fractions of the frame area. A box smaller than min_area_frac is almost
    # always a false positive on a poster/chair; larger than max_area_frac is a
    # detector meltdown covering half the room.
    min_area_frac: float = 0.0004
    max_area_frac: float = 0.55
    min_aspect: float = 0.55     # height/width. People are taller than wide;
                                 # a 0.3 box is usually a desk or a shadow.
    max_aspect: float = 6.0
    min_box_px: int = 18         # shortest side, in pixels

    # Near-identical boxes from the tracker = the same person counted twice.
    duplicate_iou: float = 0.85
    # Nested boxes are the bigger problem: YOLO often reports a torso box *and*
    # a full-body box for one seated student. Their IoU can be as low as 0.3,
    # but one is ~100% contained in the other. Intersection-over-smaller-area
    # catches that; the more confident box wins.
    containment_thresh: float = 0.80


@dataclass
class TrackConfig:
    """Multi-object tracking (Phase 9)."""

    # BoT-SORT + ReID by default. Ultralytics runs a Kalman filter inside the
    # tracker; this app runs a second one on top (see KalmanConfig) for
    # smoothing and occlusion coasting.
    tracker: str = "configs/botsort_reid.yaml"
    fallback_tracker: str = "configs/botsort_fast.yaml"   # used if ReID weights
                                                          # cannot be downloaded

    # A raw tracker ID becomes a visible "Student N" only after it has been seen
    # this many times with decent confidence. This is what keeps flickering
    # low-probability blobs out of the results.
    min_hits: int = 8
    confirm_conf: float = 0.25   # smoothed confidence required to confirm
    drop_conf: float = 0.20      # below this, an occluded student only gets
                                 # half the coasting budget before being lost
    conf_ema: float = 0.15       # smoothing factor for per-track confidence

    # Occlusion handling: how long we keep predicting a student who has no
    # detection this frame (a classmate walked in front of them).
    max_coast_frames: int = 45   # ~2.2s at 20fps - still drawn, marked predicted
    forget_after_s: float = 20.0 # after this, the student is finalized as "left"

    # Two students must never sit on one body. A track about to be confirmed is
    # discarded if it overlaps a student who is already on screen by this much
    # (or by detect.containment_thresh). This is the last line of defence
    # against an identity split when the tracker issues a second ID for someone
    # it is already tracking.
    duplicate_student_iou: float = 0.55


@dataclass
class KalmanConfig:
    """Second-stage Kalman filter: box smoothing + prediction during occlusion."""

    enabled: bool = True

    # Constant-velocity model noise, scaled by box height (DeepSORT convention).
    # Smaller position noise = smoother, lazier boxes.
    std_position: float = 0.045
    std_velocity: float = 0.007
    std_measurement: float = 0.045

    # Chi-square gate, 4 degrees of freedom. A measurement further than this
    # from the prediction is treated as an outlier: the filter coasts on its
    # own estimate instead of snapping the box across the room.
    gate_chi2: float = 9.4877
    reject_outliers: bool = True

    # Extra exponential smoothing on width/height. Box size is the noisiest
    # part of a detection and the most visible source of "breathing" boxes.
    size_ema: float = 0.25


@dataclass
class ReidConfig:
    """Re-binding a returning student to their original ID (Phase 6 + 9).

    Ultralytics hands out a brand-new raw track ID whenever it loses someone for
    too long. Without this layer a 40-minute lesson with 12 students ends up
    reporting 60 "students". This matches new raw IDs against recently lost
    students using position + appearance.
    """

    enabled: bool = True
    max_gap_s: float = 25.0          # don't rebind to someone lost long ago
    max_distance_scale: float = 2.2  # search radius = this * box height
    min_appearance: float = 0.45     # HSV histogram correlation, -1..1
    min_score: float = 0.55          # combined score needed to accept a rebind

    # Weights for the combined score (must be meaningful relative to each other)
    w_appearance: float = 0.5
    w_distance: float = 0.3
    w_iou: float = 0.2

    hist_bins: tuple[int, int] = (16, 16)   # H, S bins of the torso histogram
    hist_ema: float = 0.1                    # how fast the stored look adapts

    # A student who is merely *coasting* can also be reclaimed: if a brand-new
    # track appears right where we are predicting them, it is them, and the
    # tracker simply renumbered. Spatially much stricter than the lost-student
    # case, because a coasting student is still on screen and a different person
    # walking past must not be able to steal their identity.
    coasting_iou: float = 0.35


@dataclass
class ZoneConfig:
    """Classroom zones / ROI (Phase 3)."""

    enabled: bool = True
    # Auto-generated grid when no polygons are given. Rows are named from the
    # front of the room backwards.
    grid_rows: int = 3
    grid_cols: int = 3
    row_names: list[str] = field(default_factory=lambda: ["front", "middle", "back"])
    col_names: list[str] = field(default_factory=lambda: ["left", "center", "right"])

    # Explicit polygons win over the grid. Coordinates are normalized (0..1) so
    # the same config works on 1080p and on a 4K camera.
    #   polygons: {"door": [[0.0, 0.6], [0.15, 0.6], [0.15, 1.0], [0.0, 1.0]]}
    polygons: dict[str, list[list[float]]] = field(default_factory=dict)

    # Optional region of interest. Students whose feet fall outside it are
    # ignored entirely - useful to cut out a corridor visible through a door.
    roi: list[list[float]] = field(default_factory=list)

    draw: bool = True


@dataclass
class MotionConfig:
    """Background subtraction for movement/activity (Phase 5)."""

    enabled: bool = True
    history: int = 500
    var_threshold: float = 40.0
    detect_shadows: bool = True
    scale: float = 0.4               # run MOG2 on a downscaled frame - cheap
    open_kernel: int = 3             # morphological opening removes speckle

    # A student is "moving" when Kalman speed or foreground activity crosses
    # these thresholds for `min_frames` consecutive frames (debounce).
    # Two different kinds of movement, deliberately kept separate:
    #   speed    - the student is walking across the room
    #   activity - the student is moving in place (leaning, turning, hand up)
    # Measured on the demo footage, a seated student fidgeting peaks around
    # 0.10 foreground fraction and a still one sits near 0.00, so 0.06 splits
    # them. Raise it if ordinary desk work registers as movement.
    speed_px_per_s: float = 45.0
    activity_frac: float = 0.06
    min_frames: int = 5


@dataclass
class OutputConfig:
    """What gets written to disk for later analytics."""

    dir: str = "outputs"
    session_name: str = ""           # "" = auto timestamp

    save_video: bool = True
    video_fourcc: str = "mp4v"

    save_csv: bool = True
    save_json: bool = True
    save_sqlite: bool = True
    save_snapshots: bool = True
    save_heatmap: bool = True

    sample_every: int = 1            # write a track row every N processed frames
    trajectory_every: int = 5        # keep a trajectory point every N frames
    occupancy_bucket_s: float = 10.0 # occupancy timeline resolution
    snapshot_min_conf: float = 0.45  # crops at or above this are strongly
                                     # preferred; it is not a hard cut, so a
                                     # low-confidence student still gets a photo
    flush_every: int = 200           # DB commit interval (frames)


@dataclass
class ViewConfig:
    """Live preview and drawing (Phase 1)."""

    show: bool = True
    window: str = "Classroom Analytics"
    trail_length: int = 40           # trajectory tail, in trajectory points
    draw_trail: bool = True
    draw_hud: bool = True
    # Most fixed cameras burn a timestamp into one corner - usually the top
    # left. Move the HUD if it collides: top-left, top-right, bottom-left,
    # bottom-right.
    hud_position: str = "top-right"
    hud_opacity: float = 0.85    # the card is nearly opaque so an OSD timestamp
                                 # underneath cannot bleed through the text
    draw_predicted: bool = True      # dashed box while a student is occluded
    box_thickness: int = 2
    font_scale: float = 0.45
    progress_every: int = 50         # console progress line, in frames


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    reid: ReidConfig = field(default_factory=ReidConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    view: ViewConfig = field(default_factory=ViewConfig)

    # ---------------------------------------------------------------- helpers
    def resolve(self, relative: str) -> Path:
        """Resolve a config path relative to the project folder."""
        p = Path(relative).expanduser()
        return p if p.is_absolute() else (PROJECT_DIR / p).resolve()

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.output.dir)


def _apply(obj: Any, data: dict) -> list[str]:
    """Recursively copy known keys from `data` onto a dataclass. Returns warnings."""
    warnings: list[str] = []
    known = {f.name: f for f in fields(obj)}

    for key, value in (data or {}).items():
        if key not in known:
            warnings.append(f"unknown config key ignored: {key}")
            continue

        current = getattr(obj, key)

        if is_dataclass(current) and isinstance(value, dict):
            warnings.extend(_apply(current, value))
            continue

        # tuples in the dataclass stay tuples (YAML gives us lists)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)

        setattr(obj, key, value)

    return warnings


def load_config(path: str | Path | None = None) -> tuple[AppConfig, list[str]]:
    """Build an AppConfig from defaults + an optional YAML file."""
    config = AppConfig()
    warnings: list[str] = []

    if path:
        path = Path(path)
        if not path.is_absolute():
            path = (PROJECT_DIR / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        warnings = _apply(config, data)

    return config, warnings


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env reader (no third-party dependency).

    Existing environment variables always win, so a shell export overrides the
    file. Used to keep the RTSP password out of the repository.
    """
    env_path = Path(path) if path else PROJECT_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
