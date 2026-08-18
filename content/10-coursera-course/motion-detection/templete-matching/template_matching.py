"""
Template Matching - Simple Version

Goal:
1. Read a video.
2. Select a reference object (template) from the first frame.
3. Find that same template in every following frame.
4. Draw the detected location.
5. Save the processed video.

Run from the template-matching folder:

    python template_matching.py
"""

import cv2
from pathlib import Path

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

VIDEO_PATH = Path("assets/cup.mp4")
OUTPUT_PATH = Path("outputs/cup_template_matching.mp4")


# ---------------------------------------------------------
# 2. Open video
# ---------------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Some videos may report an invalid FPS.
if fps <= 0:
    fps = 30.0


# Create output directory if it doesn't exist.
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 3. Create video writer
# ---------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


# ---------------------------------------------------------
# 4. Read the first frame
# ---------------------------------------------------------

ret, first_frame = cap.read()

if not ret:
    cap.release()
    writer.release()

    raise RuntimeError(
        "Could not read the first frame."
    )


# ---------------------------------------------------------
# 5. Select the template manually
# ---------------------------------------------------------

print("Select the PEN with your mouse.")
print("Press ENTER or SPACE after selecting.")
print("Press C to cancel.")


# ---------------------------------------------------------
# Resize only for displaying the selection window
# ---------------------------------------------------------

display_scale = 0.5

display_frame = cv2.resize(
    first_frame,
    None,
    fx=display_scale,
    fy=display_scale
)


x, y, w, h = cv2.selectROI(
    "Select Template",
    display_frame,
    fromCenter=False,
    showCrosshair=True
)


cv2.destroyWindow("Select Template")


# ---------------------------------------------------------
# Convert ROI coordinates back to original resolution
# ---------------------------------------------------------

x = int(x / display_scale)
y = int(y / display_scale)
w = int(w / display_scale)
h = int(h / display_scale)


if w == 0 or h == 0:
    cap.release()
    writer.release()

    raise RuntimeError(
        "No template was selected."
    )


# ---------------------------------------------------------
# 6. Create template
# ---------------------------------------------------------

template = first_frame[
    y:y + h,
    x:x + w
]


# Convert template to grayscale.
template_gray = cv2.cvtColor(
    template,
    cv2.COLOR_BGR2GRAY
)


template_h, template_w = template_gray.shape


# ---------------------------------------------------------
# 7. Process video
# ---------------------------------------------------------

frame = first_frame


while True:

    # -----------------------------------------------------
    # Convert current frame to grayscale
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # Find template inside current frame
    # -----------------------------------------------------

    result = cv2.matchTemplate(
        gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )


    # -----------------------------------------------------
    # Find the best match
    # -----------------------------------------------------

    _, max_score, _, max_location = cv2.minMaxLoc(
        result
    )


    # Top-left corner of detected template
    top_left = max_location


    # Bottom-right corner
    bottom_right = (
        top_left[0] + template_w,
        top_left[1] + template_h
    )


    # -----------------------------------------------------
    # Draw bounding box
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (0, 255, 0),
        2
    )


    # -----------------------------------------------------
    # Display matching score
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Match score: {max_score:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # -----------------------------------------------------
    # Display title
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "SIMPLE TEMPLATE MATCHING",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # -----------------------------------------------------
    # Save processed frame
    # -----------------------------------------------------

    writer.write(frame)


    # -----------------------------------------------------
    # Display processed frame
    # -----------------------------------------------------

    # Resize frame only for display
    display_frame = cv2.resize(
        frame,
        None,
        fx=0.5,
        fy=0.5
    )

    cv2.imshow(
        "Simple Template Matching",
        display_frame
    )


    # -----------------------------------------------------
    # Keyboard control
    # -----------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # Press Q to quit.
    if key == ord("q"):
        break


    # -----------------------------------------------------
    # Read next frame
    # -----------------------------------------------------

    ret, frame = cap.read()


    if not ret:
        break


# ---------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------

cap.release()

writer.release()

cv2.destroyAllWindows()


print(
    f"\nSaved output video to: {OUTPUT_PATH}"
)