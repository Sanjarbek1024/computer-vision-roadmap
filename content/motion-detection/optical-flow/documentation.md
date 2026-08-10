# Optical Flow Motion Detection with OpenCV

> **Goal:** Understand and implement Optical Flow for motion detection using Python + OpenCV (`cv2`).

---

# 1. What is Optical Flow?

Optical Flow is a computer vision technique used to estimate how pixels or feature points move between two consecutive video frames.

Instead of simply asking:

> "Did the pixel change?"

Optical Flow asks:

> "Where did this point move?"

Example:

```text
Frame t                      Frame t+1

    ●                            ●
    │                            │
    │                            │
    └───────────────→            ●

       Motion Vector
```

A motion vector contains:

```text
(dx, dy)
```

where:

* `dx` → horizontal movement
* `dy` → vertical movement

The movement magnitude can be calculated as:

```text
magnitude = sqrt(dx² + dy²)
```

---

# 2. Optical Flow vs Traditional Motion Detection

Traditional frame differencing:

```text
Frame 1
   ↓
Frame 2
   ↓
Absolute Difference
   ↓
Threshold
   ↓
Motion Mask
```

Optical Flow:

```text
Frame 1
   ↓
Find feature points
   ↓
Frame 2
   ↓
Track feature points
   ↓
Motion vectors
   ↓
Direction + magnitude
   ↓
Motion analysis
```

### Frame Differencing

Good for:

* Simple motion detection
* Static camera
* Detecting changed regions

Weakness:

* Does not directly provide motion direction
* Sensitive to illumination changes
* Difficult to understand how something moved

### Optical Flow

Good for:

* Object tracking
* Motion analysis
* Camera motion estimation
* Video stabilization
* Autonomous systems
* Action analysis

---

# 3. Optical Flow Pipeline

The basic pipeline is:

```text
Video
  │
  ▼
Read Frame t
  │
  ▼
Read Frame t+1
  │
  ▼
Convert to Grayscale
  │
  ▼
Detect Feature Points
  │
  ▼
Track Points
  │
  ▼
Calculate Motion Vectors
  │
  ▼
Calculate Magnitude
  │
  ▼
Apply Motion Threshold
  │
  ▼
Visualize Motion
  │
  ▼
Save Output Video
```

---

# 4. Sparse vs Dense Optical Flow

There are two major approaches.

## Sparse Optical Flow

Only selected important points are tracked.

```text
Frame

●       ●

    ●

          ●

  ●
```

Advantages:

* Faster
* Less computationally expensive
* Good for tracking
* Easy to visualize

We will use:

```python
cv2.calcOpticalFlowPyrLK()
```

This is the **Lucas-Kanade Optical Flow** method.

---

## Dense Optical Flow

Motion is estimated for almost every pixel.

```text
→ → ↘ ↓ ↓
→ ↗ → → ↓
↑ → → ↘ →
→ → → → →
```

Advantages:

* Gives detailed motion information
* Useful for motion fields
* Better for detailed motion analysis

Disadvantages:

* More computationally expensive

Later we can learn:

```python
cv2.calcOpticalFlowFarneback()
```

---

# 5. Project Structure

For our notebook:

```text
project/
│
├── assets/
│   └── video.mp4
│
├── output/
│   └── optical_flow_output.mp4
│
└── optical_flow.ipynb
```

Input:

```text
assets/video.mp4
```

Output:

```text
output/optical_flow_output.mp4
```

---

# 6. Required Libraries

```python
import cv2
import numpy as np

from pathlib import Path
from IPython.display import Video, display
```

---

# 7. `Path` — Working with File Paths

We use `Path` instead of manually writing strings.

```python
from pathlib import Path
```

Example:

```python
VIDEO_PATH = Path("assets/video.mp4")
OUTPUT_DIR = Path("output")
OUTPUT_PATH = OUTPUT_DIR / "optical_flow_output.mp4"
```

### Why?

Instead of:

```python
"output/optical_flow_output.mp4"
```

we can work with:

```python
OUTPUT_PATH
```

This is cleaner and more portable.

---

# 8. `Path.mkdir()`

Create the output directory:

```python
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

### Parameters

```python
parents=True
```

Creates parent directories if necessary.

```python
exist_ok=True
```

Doesn't raise an error if the directory already exists.

---

# 9. `Path.exists()`

Check whether the video exists:

```python
VIDEO_PATH.exists()
```

Example:

```python
if not VIDEO_PATH.exists():
    raise FileNotFoundError("Video not found.")
```

---

# 10. Opening a Video

Use:

```python
cv2.VideoCapture()
```

Example:

```python
cap = cv2.VideoCapture(str(VIDEO_PATH))
```

### `cv2.VideoCapture()`

Used to:

* Open a video file
* Read frames
* Access video properties

---

# 11. `cap.isOpened()`

Check whether OpenCV successfully opened the video:

```python
if not cap.isOpened():
    raise RuntimeError("Could not open video.")
```

---

# 12. `cap.read()`

Read one frame:

```python
ret, frame = cap.read()
```

Returns:

```text
ret
frame
```

### `ret`

Boolean:

```text
True  → frame successfully read
False → frame unavailable
```

### `frame`

The actual image.

Typical loop:

```python
while True:

    ret, frame = cap.read()

    if not ret:
        break
```

---

# 13. Video Properties

We can get video information using:

```python
cap.get()
```

---

## FPS

```python
fps = cap.get(cv2.CAP_PROP_FPS)
```

FPS means:

> Frames Per Second

Example:

```text
30 FPS
```

means the video contains approximately 30 frames every second.

---

## Frame Count

```python
frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)
```

---

## Width

```python
width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)
```

---

## Height

```python
height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)
```

---

# 14. Release the Video

When finished:

```python
cap.release()
```

Always release the `VideoCapture`.

---

# 15. Reading Two Consecutive Frames

Optical Flow needs at least two frames:

```text
Frame t
   ↓
Frame t+1
```

Example:

```python
cap = cv2.VideoCapture(str(VIDEO_PATH))

ret1, frame1 = cap.read()
ret2, frame2 = cap.read()

cap.release()
```

---

# 16. Convert Frames to Grayscale

Optical Flow usually works on grayscale images.

```python
gray1 = cv2.cvtColor(
    frame1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    frame2,
    cv2.COLOR_BGR2GRAY
)
```

---

# 17. `cv2.cvtColor()`

General syntax:

```python
cv2.cvtColor(image, conversion_code)
```

Example:

```python
cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)
```

### Important

OpenCV uses:

```text
BGR
```

not:

```text
RGB
```

Common conversions:

```python
cv2.COLOR_BGR2GRAY
```

BGR → Grayscale

```python
cv2.COLOR_BGR2RGB
```

BGR → RGB

---

# 18. Why Grayscale?

A color image contains:

```text
Blue
Green
Red
```

Optical Flow mainly needs intensity information.

So instead of:

```text
3 channels
```

we use:

```text
1 channel
```

This makes processing simpler and faster.

---

# 19. Detect Feature Points

For Sparse Optical Flow, we need good points to track.

OpenCV provides:

```python
cv2.goodFeaturesToTrack()
```

Example:

```python
points = cv2.goodFeaturesToTrack(
    gray_frame,
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)
```

---

# 20. `cv2.goodFeaturesToTrack()`

This function detects strong corners/features.

General form:

```python
cv2.goodFeaturesToTrack(
    image,
    maxCorners,
    qualityLevel,
    minDistance,
    blockSize
)
```

---

## `maxCorners`

Maximum number of corners to return.

```python
maxCorners=300
```

Means:

> Return at most 300 feature points.

Higher:

```text
1000
```

→ more points

Lower:

```text
50
```

→ fewer points

---

## `qualityLevel`

Minimum quality of detected corners.

Example:

```python
qualityLevel=0.01
```

Lower value:

```text
more points
```

Higher value:

```text
fewer but stronger points
```

---

## `minDistance`

Minimum distance between detected points.

```python
minDistance=7
```

Prevents many points from appearing in exactly the same area.

---

## `blockSize`

Size of the neighborhood used when calculating corner quality.

```python
blockSize=7
```

---

# 21. Feature Points Structure

OpenCV normally returns points like:

```text
[
    [[x1, y1]],
    [[x2, y2]],
    [[x3, y3]]
]
```

Example:

```python
points.shape
```

may return:

```text
(300, 1, 2)
```

Meaning:

```text
300 points
1 nested dimension
2 coordinates (x, y)
```

---

# 22. Lucas-Kanade Optical Flow

The main function:

```python
cv2.calcOpticalFlowPyrLK()
```

Example:

```python
new_points, status, errors = cv2.calcOpticalFlowPyrLK(
    old_gray,
    new_gray,
    old_points,
    None
)
```

---

# 23. What Does `calcOpticalFlowPyrLK()` Do?

It answers:

> "Where did these points move in the next frame?"

Input:

```text
Old frame
Old feature points
New frame
```

Output:

```text
New feature points
Tracking status
Tracking error
```

Conceptually:

```text
Old Point
   ●
   │
   │ Optical Flow
   ↓
New Point
   ●
```

---

# 24. Parameters of `calcOpticalFlowPyrLK()`

Typical configuration:

```python
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)
```

Then:

```python
new_points, status, errors = cv2.calcOpticalFlowPyrLK(
    old_gray,
    gray,
    old_points,
    None,
    **lk_params
)
```

---

# 25. `winSize`

```python
winSize=(21, 21)
```

Defines the search window around each feature.

Larger:

```text
larger search area
```

Useful when motion is larger.

Smaller:

```text
faster
```

but may fail when points move significantly.

---

# 26. `maxLevel`

```python
maxLevel=3
```

Lucas-Kanade can use image pyramids.

Conceptually:

```text
Original image
      ↓
Smaller image
      ↓
Even smaller image
      ↓
Even smaller image
```

This helps track larger motion.

```text
maxLevel=0
```

means no pyramid levels.

```text
maxLevel=3
```

allows multiple pyramid levels.

---

# 27. `criteria`

Defines when the iterative search should stop.

```python
criteria=(
    cv2.TERM_CRITERIA_EPS |
    cv2.TERM_CRITERIA_COUNT,
    30,
    0.01
)
```

There are two conditions.

### `TERM_CRITERIA_COUNT`

Stop after a maximum number of iterations.

```text
30
```

### `TERM_CRITERIA_EPS`

Stop when the solution becomes sufficiently accurate.

```text
0.01
```

Using `|` means:

```python
EPS OR COUNT
```

---

# 28. `status`

The returned `status` tells us whether tracking succeeded.

Example:

```python
new_points, status, errors = cv2.calcOpticalFlowPyrLK(...)
```

Then:

```python
good_old = old_points[status == 1]
good_new = new_points[status == 1]
```

Meaning:

```text
status == 1
```

→ successfully tracked

```text
status == 0
```

→ tracking failed

---

# 29. `errors`

The third output:

```python
errors
```

contains tracking error information.

Example:

```python
new_points, status, errors = cv2.calcOpticalFlowPyrLK(...)
```

For basic motion detection we usually don't need it.

But it can be useful for:

* Filtering unreliable points
* Debugging
* Tracking quality analysis

---

# 30. Getting Motion Vectors

Once we have:

```python
good_old
good_new
```

we calculate:

```python
motion_vectors = good_new - good_old
```

Each vector is:

```text
(dx, dy)
```

Example:

```text
Old point = (100, 200)
New point = (108, 205)

Motion = (8, 5)
```

---

# 31. Extract `dx` and `dy`

```python
dx = motion_vectors[:, 0]
dy = motion_vectors[:, 1]
```

`dx`:

```text
horizontal movement
```

`dy`:

```text
vertical movement
```

---

# 32. Motion Magnitude

We can calculate movement size:

```python
magnitude = np.sqrt(
    dx**2 + dy**2
)
```

Equivalent mathematical idea:

```text
magnitude = √(dx² + dy²)
```

Example:

```text
dx = 3
dy = 4

magnitude = 5
```

---

# 33. NumPy `np.sqrt()`

```python
np.sqrt()
```

Calculates square root.

Example:

```python
np.sqrt(25)
```

returns:

```text
5
```

For arrays:

```python
np.sqrt(dx**2 + dy**2)
```

calculates magnitude for every motion vector.

---

# 34. Motion Threshold

Not every motion vector is meaningful.

Small movements may come from:

* Camera noise
* Compression
* Small lighting changes
* Tracking errors
* Sensor noise

So we define:

```python
MOTION_THRESHOLD = 2.0
```

Then:

```python
moving_mask = magnitude > MOTION_THRESHOLD
```

---

# 35. Boolean Mask

Example:

```python
magnitude = np.array([
    0.5,
    1.2,
    3.4,
    8.1
])
```

If:

```python
MOTION_THRESHOLD = 2
```

then:

```python
magnitude > 2
```

produces:

```text
False
False
True
True
```

This is called a **Boolean mask**.

---

# 36. Select Moving Points

```python
moving_old = good_old[moving_mask]
moving_new = good_new[moving_mask]
```

Now only significant motion remains.

---

# 37. Drawing Motion Vectors

We can draw an arrow using:

```python
cv2.arrowedLine()
```

Example:

```python
cv2.arrowedLine(
    frame,
    (x_old, y_old),
    (x_new, y_new),
    (0, 255, 0),
    2,
    tipLength=0.3
)
```

---

# 38. `cv2.arrowedLine()`

General:

```python
cv2.arrowedLine(
    image,
    start_point,
    end_point,
    color,
    thickness,
    tipLength
)
```

### `start_point`

Old location:

```python
(x_old, y_old)
```

### `end_point`

New location:

```python
(x_new, y_new)
```

### `thickness`

Arrow thickness:

```python
2
```

### `tipLength`

Arrowhead size:

```python
0.3
```

---

# 39. Drawing Feature Points

Use:

```python
cv2.circle()
```

Example:

```python
cv2.circle(
    frame,
    (x_new, y_new),
    4,
    (0, 0, 255),
    -1
)
```

---

# 40. `cv2.circle()`

General:

```python
cv2.circle(
    image,
    center,
    radius,
    color,
    thickness
)
```

Example:

```python
cv2.circle(
    frame,
    (100, 200),
    4,
    (0, 0, 255),
    -1
)
```

`-1` means:

> Fill the circle.

---

# 41. Adding Text

Use:

```python
cv2.putText()
```

Example:

```python
cv2.putText(
    frame,
    "Optical Flow",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)
```

---

# 42. `cv2.putText()`

General:

```python
cv2.putText(
    image,
    text,
    position,
    font,
    font_scale,
    color,
    thickness
)
```

Useful for displaying:

```text
Tracked points
Moving points
Average motion
Frame number
FPS
```

---

# 43. Creating the Output Video

Use:

```python
cv2.VideoWriter()
```

Example:

```python
fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)
```

---

# 44. `cv2.VideoWriter_fourcc()`

Defines the video codec.

Example:

```python
cv2.VideoWriter_fourcc(*"mp4v")
```

`mp4v` is commonly used for MP4 output.

---

# 45. `cv2.VideoWriter()`

Used to save processed frames into a video.

General:

```python
cv2.VideoWriter(
    output_path,
    codec,
    fps,
    frame_size
)
```

Example:

```python
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)
```

---

# 46. Writing a Frame

After processing:

```python
writer.write(frame)
```

This adds one frame to the output video.

---

# 47. Release the Writer

At the end:

```python
writer.release()
```

Important.

Otherwise the video may not be finalized correctly.

---

# 48. Displaying the Result in Jupyter

We can use:

```python
from IPython.display import Video, display
```

Then:

```python
display(
    Video(
        str(OUTPUT_PATH),
        embed=True
    )
)
```

---

# 49. Complete Optical Flow Algorithm

The entire algorithm is:

```text
Open video
    ↓
Read first frame
    ↓
Convert to grayscale
    ↓
Read next frame
    ↓
Convert to grayscale
    ↓
Detect feature points
    ↓
Lucas-Kanade Optical Flow
    ↓
Check tracking status
    ↓
Get old and new coordinates
    ↓
Calculate dx / dy
    ↓
Calculate motion magnitude
    ↓
Apply threshold
    ↓
Draw motion vectors
    ↓
Write output frame
    ↓
Move to next frame
    ↓
Repeat
```

---

# 50. Complete Code

```python
import cv2
import numpy as np

from pathlib import Path
from IPython.display import Video, display


# =========================================================
# 1. Paths
# =========================================================

VIDEO_PATH = Path("assets/video.mp4")

OUTPUT_DIR = Path("output")
OUTPUT_PATH = OUTPUT_DIR / "optical_flow_output.mp4"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. Check input video
# =========================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )


# =========================================================
# 3. Open video
# =========================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open video."
    )


# =========================================================
# 4. Get video properties
# =========================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# =========================================================
# 5. Create output writer
# =========================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


# =========================================================
# 6. Feature detection parameters
# =========================================================

feature_params = dict(

    maxCorners=300,

    qualityLevel=0.01,

    minDistance=7,

    blockSize=7
)


# =========================================================
# 7. Lucas-Kanade parameters
# =========================================================

lk_params = dict(

    winSize=(21, 21),

    maxLevel=3,

    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)


# =========================================================
# 8. Motion threshold
# =========================================================

MOTION_THRESHOLD = 2.0


# =========================================================
# 9. Read first frame
# =========================================================

ret, old_frame = cap.read()

if not ret:
    cap.release()
    writer.release()

    raise RuntimeError(
        "Could not read first frame."
    )


old_gray = cv2.cvtColor(
    old_frame,
    cv2.COLOR_BGR2GRAY
)


# =========================================================
# 10. Process video
# =========================================================

frame_number = 1


while True:

    # -----------------------------------------------------
    # Read next frame
    # -----------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        break


    # -----------------------------------------------------
    # Convert current frame to grayscale
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # Detect feature points
    # -----------------------------------------------------

    old_points = cv2.goodFeaturesToTrack(
        old_gray,
        mask=None,
        **feature_params
    )


    if old_points is not None:

        # -------------------------------------------------
        # Calculate Optical Flow
        # -------------------------------------------------

        new_points, status, errors = (
            cv2.calcOpticalFlowPyrLK(
                old_gray,
                gray,
                old_points,
                None,
                **lk_params
            )
        )


        if new_points is not None:

            # ---------------------------------------------
            # Keep successfully tracked points
            # ---------------------------------------------

            good_old = old_points[
                status == 1
            ]

            good_new = new_points[
                status == 1
            ]


            # ---------------------------------------------
            # Calculate motion vectors
            # ---------------------------------------------

            vectors = (
                good_new - good_old
            )


            # ---------------------------------------------
            # Extract dx and dy
            # ---------------------------------------------

            dx = vectors[:, 0]

            dy = vectors[:, 1]


            # ---------------------------------------------
            # Calculate magnitude
            # ---------------------------------------------

            magnitude = np.sqrt(
                dx**2 + dy**2
            )


            # ---------------------------------------------
            # Filter significant motion
            # ---------------------------------------------

            moving_mask = (
                magnitude >
                MOTION_THRESHOLD
            )


            moving_old = (
                good_old[moving_mask]
            )

            moving_new = (
                good_new[moving_mask]
            )


            # ---------------------------------------------
            # Draw motion vectors
            # ---------------------------------------------

            for old, new in zip(
                moving_old,
                moving_new
            ):

                x_old, y_old = (
                    old.ravel()
                )

                x_new, y_new = (
                    new.ravel()
                )


                cv2.arrowedLine(
                    frame,

                    (
                        int(x_old),
                        int(y_old)
                    ),

                    (
                        int(x_new),
                        int(y_new)
                    ),

                    (0, 255, 0),

                    2,

                    tipLength=0.3
                )


                cv2.circle(
                    frame,

                    (
                        int(x_new),
                        int(y_new)
                    ),

                    3,

                    (0, 0, 255),

                    -1
                )


            # ---------------------------------------------
            # Calculate statistics
            # ---------------------------------------------

            avg_motion = (
                magnitude.mean()
            )


            # ---------------------------------------------
            # Display statistics
            # ---------------------------------------------

            cv2.putText(
                frame,

                f"Tracked points: {len(good_old)}",

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2
            )


            cv2.putText(
                frame,

                f"Moving points: {len(moving_old)}",

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )


            cv2.putText(
                frame,

                f"Avg motion: {avg_motion:.2f}px",

                (20, 105),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 255),

                2
            )


    # -----------------------------------------------------
    # Write processed frame
    # -----------------------------------------------------

    writer.write(
        frame
    )


    # -----------------------------------------------------
    # Current frame becomes previous frame
    # -----------------------------------------------------

    old_gray = gray


    frame_number += 1


    if frame_number % 30 == 0:

        print(
            f"Processed {frame_number} frames..."
        )


# =========================================================
# 11. Release resources
# =========================================================

cap.release()

writer.release()


print()
print("Processing completed.")
print(
    f"Output saved to: {OUTPUT_PATH}"
)
```

---

# 51. Display Output

```python
display(
    Video(
        str(OUTPUT_PATH),
        embed=True
    )
)
```

---

# 52. Important Parameters to Experiment With

The most important parameters are:

```python
maxCorners=300
qualityLevel=0.01
minDistance=7
```

and:

```python
winSize=(21, 21)
maxLevel=3
```

and:

```python
MOTION_THRESHOLD=2.0
```

---

# 53. Experiment: More Feature Points

Try:

```python
maxCorners=500
```

Compared with:

```python
maxCorners=100
```

Expected result:

```text
100 points
→ faster
→ fewer motion vectors

500 points
→ more detailed
→ more computationally expensive
```

---

# 54. Experiment: Motion Threshold

Try:

```python
MOTION_THRESHOLD = 1.0
```

Then:

```python
MOTION_THRESHOLD = 5.0
```

Then:

```python
MOTION_THRESHOLD = 10.0
```

Lower threshold:

```text
more points classified as moving
```

Higher threshold:

```text
only stronger movement
```

---

# 55. Important Problem: Camera Movement

Optical Flow detects **image motion**, not automatically object motion.

Suppose:

```text
Camera moves →
```

Then the entire background may produce motion vectors:

```text
→ → → → → → →
→ → → → → → →
→ → → → → → →
```

Even if the person is standing still.

This is one of the most important concepts in Optical Flow.

---

# 56. Camera Motion vs Object Motion

Imagine:

```text
Camera
  │
  ▼
──────────────
Background
      Person
        ●
```

If the camera moves:

```text
Background  → → →
Person      → → →
```

Optical Flow sees motion everywhere.

But what we actually want may be:

```text
Camera motion
      ↓
remove it
      ↓
remaining motion
      ↓
Object motion
```

This becomes important in:

* Video stabilization
* Object tracking
* Autonomous driving
* Surveillance
* Action recognition

---

# 57. Optical Flow for Video Stabilization

The same optical flow information can estimate camera movement.

Pipeline:

```text
Video
  ↓
Feature Detection
  ↓
Optical Flow
  ↓
Track Features
  ↓
Estimate Global Motion
  ↓
Camera Motion
  ↓
Smooth Motion
  ↓
Warp Frames
  ↓
Stabilized Video
```

This is why Optical Flow is directly connected to our previous stabilization work.

---

# 58. Limitations of Lucas-Kanade

Lucas-Kanade is powerful but not perfect.

Problems include:

### Large motion

If a point moves too far:

```text
● ------------------------------→ ●
```

tracking can fail.

Possible solution:

```python
maxLevel=3
```

and larger:

```python
winSize=(31, 31)
```

---

### Occlusion

A feature can disappear behind another object.

```text
Frame 1:

●


Frame 2:

████
```

The original point cannot be tracked.

---

### Motion Blur

Fast movement can create:

```text
████████
```

instead of a clear feature.

Tracking becomes harder.

---

### Low Texture

Flat surfaces are difficult to track.

Example:

```text
████████████████
████████████████
████████████████
```

There may be very few good corners.

---

# 59. Important OpenCV Functions

For this notebook, remember these functions:

| Function                     | Purpose                       |
| ---------------------------- | ----------------------------- |
| `cv2.VideoCapture()`         | Open video                    |
| `cap.read()`                 | Read frame                    |
| `cap.get()`                  | Get video properties          |
| `cap.release()`              | Release video                 |
| `cv2.cvtColor()`             | Convert color space           |
| `cv2.goodFeaturesToTrack()`  | Detect feature points         |
| `cv2.calcOpticalFlowPyrLK()` | Calculate sparse optical flow |
| `cv2.arrowedLine()`          | Draw motion vector            |
| `cv2.circle()`               | Draw point                    |
| `cv2.putText()`              | Draw text                     |
| `cv2.VideoWriter()`          | Create output video           |
| `cv2.VideoWriter_fourcc()`   | Select codec                  |
| `writer.write()`             | Save frame                    |
| `writer.release()`           | Close output video            |

---

# 60. Mental Model

The easiest way to remember Optical Flow:

```text
1. Find interesting points
          ↓
2. Find those points in next frame
          ↓
3. Compare old position vs new position
          ↓
4. Calculate movement
          ↓
5. Analyze movement
```

In code:

```python
points = cv2.goodFeaturesToTrack(...)
```

↓

```python
new_points, status, errors = \
    cv2.calcOpticalFlowPyrLK(...)
```

↓

```python
vectors = good_new - good_old
```

↓

```python
magnitude = np.sqrt(dx**2 + dy**2)
```

↓

```python
moving_mask = magnitude > threshold
```

---

# 61. The Core Code to Remember

If you forget everything else, remember this:

```python
# Detect points
old_points = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=7
)

# Track points
new_points, status, errors = \
    cv2.calcOpticalFlowPyrLK(
        old_gray,
        gray,
        old_points,
        None
    )

# Keep successful tracks
good_old = old_points[
    status == 1
]

good_new = new_points[
    status == 1
]

# Motion vectors
vectors = (
    good_new - good_old
)

# Motion magnitude
magnitude = np.sqrt(
    vectors[:, 0]**2 +
    vectors[:, 1]**2
)
```

This is the **core of Sparse Optical Flow**.

---

# 62. Next Level

After understanding this notebook, the natural progression is:

```text
OPTICAL FLOW
│
├── 1. Sparse Optical Flow
│      │
│      └── Lucas-Kanade
│
├── 2. Dense Optical Flow
│      │
│      └── Farneback
│
├── 3. Motion Visualization
│
├── 4. Motion Detection
│
├── 5. Object Tracking
│
├── 6. Camera Motion Estimation
│
└── 7. Video Stabilization
```

The key transition is:

```text
"Where are points moving?"
           ↓
"How is the whole image moving?"
           ↓
"Which motion belongs to the camera?"
           ↓
"Which motion belongs to the object?"
```

That is where Optical Flow becomes much more powerful than simple frame differencing.
