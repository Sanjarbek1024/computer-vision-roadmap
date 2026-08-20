"""Classroom Analytics - the final demo project of the Computer Vision roadmap.

Person detection (YOLO11-nano) + multi-object tracking (BoT-SORT + ReID) +
a second-stage Kalman filter, wired into a pipeline that produces analytics a
school can actually use: attendance, time in room, zone occupancy, movement,
and an event log, all stored for later.

Typical use:

    from classroom_analytics import AppConfig, ClassroomPipeline

    cfg = AppConfig()
    cfg.source.path = "class_videos/lesson.mp4"
    ClassroomPipeline(cfg).run()
"""

from .analytics import SessionAnalytics
from .config import AppConfig, load_config, load_dotenv
from .detector import Detection, PersonDetector
from .kalman import BoxKalman
from .motion import MotionAnalyzer
from .pipeline import ClassroomPipeline
from .preprocess import Preprocessor
from .registry import Student, StudentRegistry
from .storage import SessionStore
from .video_source import VideoSource
from .visualizer import Visualizer
from .zones import ZoneMap

__version__ = "1.0.0"

__all__ = [
    "AppConfig",
    "BoxKalman",
    "ClassroomPipeline",
    "Detection",
    "MotionAnalyzer",
    "PersonDetector",
    "Preprocessor",
    "SessionAnalytics",
    "SessionStore",
    "Student",
    "StudentRegistry",
    "VideoSource",
    "Visualizer",
    "ZoneMap",
    "load_config",
    "load_dotenv",
]
