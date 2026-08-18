"""
Robust Template Matching

This version improves the basic template matching approach by:

1. Restricting the search to a moving ROI.
2. Updating the template after every successful match.
3. Updating the ROI around the newly detected object.
4. Reusing the current object position for the next frame.
5. Using a confidence threshold to reduce incorrect matches.

Important:
Updating the template every frame can cause "template drift"
if the matcher makes a wrong prediction.

Run from the template-matching folder:

    python robust_template_matching.py
"""

import cv2
from pathlib import Path


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

VIDEO_PATH = Path("assets/cup.mp4")
OUTPUT_PATH = Path("outputs/cup_robust_template_matching.mp4")

# Minimum matching score required to accept a match.
MATCH_THRESHOLD = 0.55

# Search area around the previous object position.
#
# Larger values:
#   + handle faster movement
#   - increase computation
#   - increase chance of wrong matches
#
ROI_MARGIN_X = 120
ROI_MARGIN_Y = 120


# Display scale.
#
# This affects only the window shown on the screen.
# Processing and saved video remain at original resolution.

DISPLAY_SCALE = 0.5


# ---------------------------------------------------------
# 2. Helper function: create a safe ROI
# ---------------------------------------------------------

def make_search_roi(
    frame_width,
    frame_height,
    x,
    y,
    w,
    h
):
    """
    Create an ROI around the previous object position.

    The ROI is clipped so it never goes outside
    the frame boundaries.
    """

    x1 = max(
        0,
        x - ROI_MARGIN_X
    )

    y1 = max(
        0,
        y - ROI_MARGIN_Y
    )

    x2 = min(
        frame_width,
        x + w + ROI_MARGIN_X
    )

    y2 = min(
        frame_height,
        y + h + ROI_MARGIN_Y
    )

    return x1, y1, x2, y2


# ---------------------------------------------------------
# 3. Open video
# ---------------------------------------------------------

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ---------------------------------------------------------
# 4. Get video information
# ---------------------------------------------------------

fps = cap.get(
    cv2.CAP_PROP_FPS
)


frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


# Some videos may report an invalid FPS.
if fps <= 0:

    fps = 30.0


# ---------------------------------------------------------
# 5. Create output directory
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# 6. Create video writer
# ---------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (
        frame_width,
        frame_height
    )
)


# ---------------------------------------------------------
# 7. Read first frame
# ---------------------------------------------------------

ret, frame = cap.read()


if not ret:

    cap.release()
    writer.release()

    raise RuntimeError(
        "Could not read the first frame."
    )


# ---------------------------------------------------------
# 8. Select initial template
# ---------------------------------------------------------

print(
    "Select the PEN with your mouse."
)

print(
    "Press ENTER or SPACE after selecting."
)

print(
    "Press C to cancel."
)


# ---------------------------------------------------------
# Resize first frame ONLY for ROI selection
# ---------------------------------------------------------

display_frame = cv2.resize(
    frame,
    None,
    fx=DISPLAY_SCALE,
    fy=DISPLAY_SCALE
)


x, y, w, h = cv2.selectROI(
    "Select Initial Template",
    display_frame,
    fromCenter=False,
    showCrosshair=True
)


cv2.destroyWindow(
    "Select Initial Template"
)


# ---------------------------------------------------------
# Convert ROI coordinates back to original resolution
# ---------------------------------------------------------

x = int(
    x / DISPLAY_SCALE
)

y = int(
    y / DISPLAY_SCALE
)

w = int(
    w / DISPLAY_SCALE
)

h = int(
    h / DISPLAY_SCALE
)


# ---------------------------------------------------------
# Check template selection
# ---------------------------------------------------------

if w == 0 or h == 0:

    cap.release()
    writer.release()

    raise RuntimeError(
        "No template was selected."
    )


# ---------------------------------------------------------
# 9. Create initial template
# ---------------------------------------------------------

template = frame[
    y:y + h,
    x:x + w
].copy()


# Convert template to grayscale.
template_gray = cv2.cvtColor(
    template,
    cv2.COLOR_BGR2GRAY
)


# Template dimensions.
template_h, template_w = (
    template_gray.shape
)


# Current object position.
current_x = x
current_y = y


# ---------------------------------------------------------
# 10. Process video
# ---------------------------------------------------------

while True:

    # -----------------------------------------------------
    # A. Create moving ROI
    # -----------------------------------------------------

    roi_x1, roi_y1, roi_x2, roi_y2 = (
        make_search_roi(
            frame_width,
            frame_height,
            current_x,
            current_y,
            template_w,
            template_h
        )
    )


    # -----------------------------------------------------
    # B. Crop search region
    # -----------------------------------------------------

    search_region = frame[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ]


    # Convert search region to grayscale.
    search_gray = cv2.cvtColor(
        search_region,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # C. Check ROI size
    # -----------------------------------------------------

    if (
        search_gray.shape[0] >= template_h
        and
        search_gray.shape[1] >= template_w
    ):

        # -------------------------------------------------
        # D. Template Matching
        # -------------------------------------------------

        result = cv2.matchTemplate(
            search_gray,
            template_gray,
            cv2.TM_CCOEFF_NORMED
        )


        # -------------------------------------------------
        # E. Find best match
        # -------------------------------------------------

        (
            _,
            max_score,
            _,
            max_location
        ) = cv2.minMaxLoc(
            result
        )


        # -------------------------------------------------
        # F. Convert ROI coordinates
        #    to full-frame coordinates
        # -------------------------------------------------

        detected_x = (
            roi_x1 + max_location[0]
        )

        detected_y = (
            roi_y1 + max_location[1]
        )


        # -------------------------------------------------
        # G. Check confidence
        # -------------------------------------------------

        if max_score >= MATCH_THRESHOLD:

            # ---------------------------------------------
            # Successful match
            # ---------------------------------------------

            current_x = detected_x
            current_y = detected_y


            status = "TRACKING"

            status_color = (
                0,
                255,
                0
            )


            # ---------------------------------------------
            # H. Extract new template
            # ---------------------------------------------

            new_template = frame[
                current_y:
                current_y + template_h,

                current_x:
                current_x + template_w
            ]


            # ---------------------------------------------
            # I. Validate new template
            # ---------------------------------------------

            if (
                new_template.shape[0]
                == template_h
                and
                new_template.shape[1]
                == template_w
            ):

                # Update template.
                template = (
                    new_template.copy()
                )


                # Convert updated template
                # to grayscale.
                template_gray = cv2.cvtColor(
                    template,
                    cv2.COLOR_BGR2GRAY
                )


        else:

            # ---------------------------------------------
            # Match rejected
            # ---------------------------------------------

            status = "LOW CONFIDENCE"

            status_color = (
                0,
                0,
                255
            )


        # -------------------------------------------------
        # J. Draw object bounding box
        # -------------------------------------------------

        cv2.rectangle(
            frame,

            (
                current_x,
                current_y
            ),

            (
                current_x + template_w,
                current_y + template_h
            ),

            status_color,

            2
        )


        # -------------------------------------------------
        # K. Draw current ROI
        # -------------------------------------------------

        cv2.rectangle(
            frame,

            (
                roi_x1,
                roi_y1
            ),

            (
                roi_x2,
                roi_y2
            ),

            (
                255,
                0,
                0
            ),

            1
        )


        # -------------------------------------------------
        # L. Calculate object center
        # -------------------------------------------------

        center_x = (
            current_x
            + template_w // 2
        )


        center_y = (
            current_y
            + template_h // 2
        )


        # -------------------------------------------------
        # M. Draw object center
        # -------------------------------------------------

        cv2.circle(
            frame,

            (
                center_x,
                center_y
            ),

            4,

            (
                0,
                255,
                255
            ),

            -1
        )


        # -------------------------------------------------
        # N. Display matching score
        # -------------------------------------------------

        cv2.putText(
            frame,

            f"Score: {max_score:.2f}",

            (
                20,
                35
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            status_color,

            2
        )


        # -------------------------------------------------
        # O. Display position
        # -------------------------------------------------

        cv2.putText(
            frame,

            (
                f"Position: "
                f"({current_x}, {current_y})"
            ),

            (
                20,
                65
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                255,
                255,
                255
            ),

            2
        )


        # -------------------------------------------------
        # P. Display status
        # -------------------------------------------------

        cv2.putText(
            frame,

            f"Status: {status}",

            (
                20,
                95
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            status_color,

            2
        )


        # -------------------------------------------------
        # Q. Display title
        # -------------------------------------------------

        cv2.putText(
            frame,

            "ROBUST TEMPLATE MATCHING",

            (
                20,
                130
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                255,
                255,
                255
            ),

            2
        )


    # -----------------------------------------------------
    # R. Save original-resolution frame
    # -----------------------------------------------------

    writer.write(
        frame
    )


    # -----------------------------------------------------
    # S. Resize ONLY for display
    # -----------------------------------------------------

    display_frame = cv2.resize(
        frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE
    )


    # -----------------------------------------------------
    # T. Display processed frame
    # -----------------------------------------------------

    cv2.imshow(
        "Robust Template Matching",
        display_frame
    )


    # -----------------------------------------------------
    # U. Keyboard control
    # -----------------------------------------------------

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # Press Q to quit.
    if key == ord("q"):

        break


    # -----------------------------------------------------
    # V. Read next frame
    # -----------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        break


# ---------------------------------------------------------
# 11. Cleanup
# ---------------------------------------------------------

cap.release()

writer.release()

cv2.destroyAllWindows()


print(
    f"\nSaved output video to: "
    f"{OUTPUT_PATH}"
)