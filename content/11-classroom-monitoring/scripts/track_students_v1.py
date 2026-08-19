import cv2
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = Path(
    "content/11-classroom-monitoring/class_videos/classroom_camera.mp4"
)

OUTPUT_PATH = Path(
    "content/11-classroom-monitoring/outputs/classroom_tracking.mp4"
)

TRACKER_CONFIG = Path(
    "content/11-classroom-monitoring/scripts/botsort_classroom.yaml"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

model = YOLO("yolo11x.pt")


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps:.2f}")


# ============================================================
# OUTPUT
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# STUDENT ID MAPPING
# ============================================================

track_to_student = {}
next_student_id = 1


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker=str(TRACKER_CONFIG),

        # Person class only
        classes=[0],

        # Lower threshold to improve small/background detection
        conf=0.15,

        # Higher resolution helps small students
        imgsz=1280,

        verbose=False
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # ------------------------------------------------
            # Assign our own Student ID
            # ------------------------------------------------

            if track_id not in track_to_student:

                track_to_student[track_id] = next_student_id
                next_student_id += 1

            student_id = track_to_student[track_id]

            label = f"Student {student_id}"

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # ------------------------------------------------
            # Small readable label
            # ------------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_y = max(y1 - 6, text_h + 4)

            cv2.rectangle(
                frame,
                (x1, label_y - text_h - baseline - 4),
                (x1 + text_w + 6, label_y + 2),
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 3, label_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )


    # ========================================================
    # DISPLAY
    # ========================================================

    writer.write(frame)

    cv2.imshow(
        "Classroom Student Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()

print()
print("Tracking completed.")
print(f"Output: {OUTPUT_PATH}")