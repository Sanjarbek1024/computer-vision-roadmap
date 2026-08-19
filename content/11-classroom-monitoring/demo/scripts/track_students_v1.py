import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


BASE = Path("content/11-classroom-monitoring/demo")
VIDEO = BASE / "class_videos/class_video.mp4"
TRACKER = BASE / "scripts/botsort_classroom.yaml"
OUTPUT = BASE / "outputs"

VIDEO_OUT = OUTPUT / "classroom_tracking.mp4"
JSON_OUT = OUTPUT / "students.json"
CSV_OUT = OUTPUT / "tracks.csv"
SNAPSHOT_DIR = OUTPUT / "students"

MODEL_NAME = "yolo11s.pt"
CONF = 0.25
IMGSZ = 960

OUTPUT.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def get_student(track_id, students, next_id):
    track_id = int(track_id)

    if track_id not in students:
        students[track_id] = {
            "student_id": int(next_id),
            "track_id": track_id,
            "first_seen": None,
            "last_seen": None,
            "frames": 0,
            "snapshot": None,
            "trajectory": []
        }
        next_id += 1

    return students[track_id], next_id


def save_snapshot(frame, box, student):
    if student["snapshot"]:
        return

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return

    folder = SNAPSHOT_DIR / f"student_{student['student_id']:02d}"
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "snapshot.jpg"

    cv2.imwrite(str(path), crop)

    student["snapshot"] = str(path)


def draw_student(frame, box, student_id):
    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        1
    )

    label = f"Student {student_id:02d}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(
        label,
        font,
        scale,
        thickness
    )

    label_x = x1
    label_y = y1 - 3

    if label_y - th - baseline < 0:
        label_y = y1 + th + baseline + 3

    cv2.rectangle(
        frame,
        (label_x, label_y - th - baseline),
        (label_x + tw + 4, label_y + 1),
        (255, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (label_x + 2, label_y - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

fps = float(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(VIDEO_OUT),
    fourcc,
    fps,
    (width, height)
)

students = {}
next_student_id = 1
frame_number = 0

csv_file = open(
    CSV_OUT,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "time_seconds",
    "student_id",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y"
])


while True:
    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker=str(TRACKER),
        classes=[0],
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False
    )

    result = results[0]

    current_detections = []

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            if x2 <= x1 or y2 <= y1:
                continue

            student, next_student_id = get_student(
                track_id,
                students,
                next_student_id
            )

            student_id = int(student["student_id"])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if student["first_seen"] is None:
                student["first_seen"] = int(frame_number)

            student["last_seen"] = int(frame_number)
            student["frames"] += 1

            student["trajectory"].append({
                "frame": int(frame_number),
                "x": cx,
                "y": cy
            })

            if len(student["trajectory"]) > 300:
                student["trajectory"].pop(0)

            save_snapshot(
                frame,
                (x1, y1, x2, y2),
                student
            )

            csv_writer.writerow([
                int(frame_number),
                round(frame_number / fps, 3),
                student_id,
                track_id,
                x1,
                y1,
                x2,
                y2,
                cx,
                cy
            ])

            current_detections.append(
                ((x1, y1, x2, y2), student_id)
            )

    for box, student_id in current_detections:
        draw_student(
            frame,
            box,
            student_id
        )

    cv2.putText(
        frame,
        f"Students: {len(students)}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    writer.write(frame)

    cv2.imshow(
        "Classroom Student Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()


student_data = []

for student in students.values():
    first_seen = int(student["first_seen"])
    last_seen = int(student["last_seen"])

    student_data.append({
        "student_id": int(student["student_id"]),
        "track_id": int(student["track_id"]),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "frames": int(student["frames"]),
        "duration_seconds": round(
            (last_seen - first_seen) / fps,
            2
        ),
        "snapshot": student["snapshot"],
        "trajectory": [
            {
                "frame": int(point["frame"]),
                "x": int(point["x"]),
                "y": int(point["y"])
            }
            for point in student["trajectory"]
        ]
    })


student_data.sort(
    key=lambda x: x["student_id"]
)


data = {
    "video": str(VIDEO),
    "model": MODEL_NAME,
    "confidence": CONF,
    "image_size": IMGSZ,
    "fps": fps,
    "resolution": [width, height],
    "total_frames": total_frames,
    "students_detected": len(student_data),
    "students": student_data
}


with open(
    JSON_OUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        data,
        f,
        indent=2
    )


print(f"Video: {VIDEO}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps:.2f}")
print(f"Frames: {total_frames}")
print(f"Students: {len(student_data)}")
print(f"Output: {VIDEO_OUT}")
print(f"JSON: {JSON_OUT}")
print(f"CSV: {CSV_OUT}")