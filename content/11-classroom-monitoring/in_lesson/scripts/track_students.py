"""
Classroom student tracking: YOLO (person detection) + BoT-SORT/ByteTrack (identity).
Tracking only - no behavior analysis. Every visible student gets a persistent
"Student N" label that stays fixed for as long as the tracker can hold the track.

Run:
    python track_students.py                  # BoT-SORT + ReID (default)
    python track_students.py --tracker bytetrack   # for comparison
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------- paths ----------------
# Adjust ROOT_DIR / MODEL_PATH if your folder layout differs from this guess:
#   <ROOT_DIR>/yolo11m.pt
#   <ROOT_DIR>/content/11-classroom-monitoring/{class_videos,outputs,scripts}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent.parent

VIDEO_PATH = PROJECT_DIR / "class_videos" / "classroom_camera.mp4"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "classroom_tracked.mp4"
MODEL_PATH = ROOT_DIR / "yolo11m.pt"

# ---------------- detection / tracking ----------------
IMG_SIZE = 1280      # up from the default 640 -> much better recall on small/background students
CONF_THRESH = 0.1    # deliberately low: don't pre-filter before the tracker's own 2-stage matching
IOU_THRESH = 0.7     # NMS overlap threshold - keep permissive so adjacent seated students aren't merged
                      # (higher = only near-duplicate boxes get suppressed, real neighbors both survive)

# ---------------- visualization ----------------
SHOW_PREVIEW = True   # live cv2 window while processing; turn off for faster full-video runs
BOX_THICKNESS = 1
FONT_SCALE = 0.45
COLORS = [
    (66, 135, 245), (52, 199, 89), (255, 149, 0), (255, 45, 85),
    (175, 82, 222), (90, 200, 250), (255, 214, 10), (162, 132, 94),
]

PROGRESS_EVERY = 100  # print a progress line every N frames


class StudentIDMapper:
    """Gives every raw tracker ID a permanent, human-readable 'Student N' label.

    This does NOT fix tracker-level identity switches by itself - that's the
    tracker's job (ReID + track_buffer, tuned in botsort_reid.yaml). This class
    just guarantees that once a raw ID is labeled, it keeps that label for the
    rest of the video instead of showing a raw tracker number.
    """

    def __init__(self):
        self._labels: dict[int, str] = {}
        self._next_num = 1

    def label_for(self, track_id: int) -> str:
        if track_id not in self._labels:
            self._labels[track_id] = f"Student {self._next_num}"
            self._next_num += 1
        return self._labels[track_id]

    @property
    def active_count(self) -> int:
        return len(self._labels)


def color_for(track_id: int) -> tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]


def draw_box(frame, xyxy, label: str, color: tuple[int, int, int]) -> None:
    """Thin box + small filled label, no confidence score."""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def run(tracker_config: Path) -> None:
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    id_mapper = StudentIDMapper()
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            tracker=str(tracker_config),
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            classes=[0],       # person only
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(xyxy, ids):
                label = id_mapper.label_for(track_id)
                draw_box(frame, box, label, color_for(track_id))

        writer.write(frame)

        if SHOW_PREVIEW:
            cv2.imshow("Classroom Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % PROGRESS_EVERY == 0:
            print(f"frame {frame_idx}/{total_frames} | students so far: {id_mapper.active_count}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done: {id_mapper.active_count} students tracked. Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort",
                         help="botsort = BoT-SORT+ReID (default), bytetrack = for comparison")
    args = parser.parse_args()

    tracker_file = SCRIPT_DIR / ("botsort_reid.yaml" if args.tracker == "botsort" else "bytetrack_tuned.yaml")
    run(tracker_file)
