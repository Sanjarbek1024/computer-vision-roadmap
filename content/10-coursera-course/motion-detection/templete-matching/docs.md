# Template Matching — OpenCV Cheatsheet

> A practical, step-by-step guide to understanding Template Matching with OpenCV.

Template Matching is a classical Computer Vision technique used to find a specific object or pattern inside an image or video.

The basic idea:

**Template + Image → Compare → Find the Best Match → Get Location**

---

## 1. The Big Picture

```text
Video
  ↓
Read Frame
  ↓
Select / Crop Template
  ↓
Convert to Grayscale
  ↓
Template Matching
  ↓
Matching Score Matrix
  ↓
Find Best Match
  ↓
Get Object Location
  ↓
Draw Bounding Box
  ↓
Repeat for Every Frame
```

For a video:

```text
Frame 1 → Match → Location
Frame 2 → Match → Location
Frame 3 → Match → Location
   ↓
   ...
Frame N → Match → Location
```

---

# 2. Import OpenCV

```python
import cv2
```

OpenCV provides the Computer Vision functions used in this project.

Check the installed version:

```python
print(cv2.__version__)
```

---

# 3. Import NumPy

```python
import numpy as np
```

Images loaded by OpenCV are represented as NumPy arrays.

Check:

```python
type(frame)
```

Usually:

```text
numpy.ndarray
```

---

# 4. Import Path

```python
from pathlib import Path
```

`Path` makes working with file paths easier.

Example:

```python
VIDEO_PATH = Path("assets/cup.mp4")
```

---

# 5. Open a Video

```python
cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)
```

`VideoCapture` creates an object that allows us to read the video frame by frame.

Check whether the video was opened successfully:

```python
cap.isOpened()
```

Expected:

```text
True
```

---

# 6. Read a Frame

```python
ret, frame = cap.read()
```

Two values are returned:

```text
ret
 ↓
Was the frame successfully read?

frame
 ↓
The actual image
```

Check:

```python
print(ret)
print(type(frame))
```

---

# 7. Understand Image Shape

```python
print(frame.shape)
```

Example:

```text
(720, 1280, 3)
```

Meaning:

```text
720  → Height
1280 → Width
3    → Color channels
```

OpenCV normally stores color images as:

```text
BGR
```

not RGB.

---

# 8. Get Video Information

### FPS

```python
fps = cap.get(
    cv2.CAP_PROP_FPS
)
```

### Width

```python
width = cap.get(
    cv2.CAP_PROP_FRAME_WIDTH
)
```

### Height

```python
height = cap.get(
    cv2.CAP_PROP_FRAME_HEIGHT
)
```

Convert them to integers:

```python
width = int(width)
height = int(height)
```

---

# 9. Template

A **template** is the small image containing the object we want to find.

```text
Full Frame
┌───────────────────────────────┐
│                               │
│       ┌───────────┐           │
│       │    CUP    │ ← Template│
│       └───────────┘           │
│                               │
└───────────────────────────────┘
```

The template is extracted from the first frame.

---

# 10. Template Coordinates

A rectangular region can be described using:

```python
x
y
w
h
```

Where:

```text
x → left position
y → top position
w → width
h → height
```

Example:

```python
x = 100
y = 150
w = 170
h = 220
```

---

# 11. Extract the Template

Images are NumPy arrays, so we can crop them using slicing:

```python
template = frame[
    y:y+h,
    x:x+w
]
```

Conceptually:

```text
Full Frame
     ↓
[y:y+h, x:x+w]
     ↓
Template
```

Check:

```python
print(template.shape)
```

---

# 12. BGR → Grayscale

Template Matching can be performed using grayscale images.

Convert the template:

```python
template_gray = cv2.cvtColor(
    template,
    cv2.COLOR_BGR2GRAY
)
```

Convert the current frame:

```python
gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY
)
```

Before:

```text
(height, width, 3)
```

After:

```text
(height, width)
```

The three color channels are reduced to one intensity channel.

---

# 13. Template Dimensions

```python
template_h, template_w = template_gray.shape
```

Now:

```text
template_h → template height
template_w → template width
```

These dimensions are later used to create the bounding box.

---

# 14. The Core — `matchTemplate()`

This is the main Template Matching operation:

```python
result = cv2.matchTemplate(
    gray,
    template_gray,
    cv2.TM_CCOEFF_NORMED
)
```

Conceptually, OpenCV slides the template across the image and compares it with each possible region.

```text
Image

┌───────────────────────────────┐
│                               │
│   [template]                  │
│       ↓                       │
│     compare                  │
│       ↓                       │
│      move                     │
│       ↓                       │
│     compare                  │
│       ↓                       │
│      ...                      │
└───────────────────────────────┘
```

The output is a **matching score matrix**.

---

# 15. `TM_CCOEFF_NORMED`

We use:

```python
cv2.TM_CCOEFF_NORMED
```

It produces normalized matching scores.

Conceptually:

```text
-1 ───────────── 0 ───────────── 1
bad                         strong
```

A score close to `1` means the region is highly similar to the template.

Example:

```text
0.21 → weak
0.54 → moderate
0.78 → good
0.94 → very strong
```

---

# 16. Inspect the Result

Check the result type:

```python
print(type(result))
```

Check the result shape:

```python
print(result.shape)
```

Check the minimum score:

```python
print(result.min())
```

Check the maximum score:

```python
print(result.max())
```

The result is **not the detected object**.

It is a score map:

```text
Result Matrix

┌───────────────────────────┐
│ 0.12  0.20  0.31  0.18   │
│ 0.10  0.42  0.71  0.22   │
│ 0.15  0.34  0.94  0.25 ← │ BEST
│ 0.11  0.20  0.30  0.17   │
└───────────────────────────┘
```

---

# 17. Find the Best Match

Use:

```python
min_value, max_value, min_location, max_location = cv2.minMaxLoc(
    result
)
```

For `TM_CCOEFF_NORMED`, we care about:

```python
max_value
max_location
```

So:

```python
max_score = max_value
top_left = max_location
```

Example:

```text
max_score = 0.94
top_left  = (320, 180)
```

Meaning:

> The best matching region starts at `(320, 180)`.

---

# 18. Bounding Box

We already know:

```python
top_left = (x, y)
```

And:

```python
template_w
template_h
```

Therefore:

```python
bottom_right = (
    top_left[0] + template_w,
    top_left[1] + template_h
)
```

Visually:

```text
top_left
   ↓
   ┌──────────────────┐
   │                  │
   │     OBJECT       │
   │                  │
   └──────────────────┘
                      ↑
                bottom_right
```

---

# 19. Draw the Detection

Create a copy of the frame:

```python
output = frame.copy()
```

Draw the bounding box:

```python
cv2.rectangle(
    output,
    top_left,
    bottom_right,
    (0, 255, 0),
    2
)
```

Arguments:

```text
image
top-left
bottom-right
color
thickness
```

---

# 20. Display the Match Score

```python
cv2.putText(
    output,
    f"Match score: {max_score:.2f}",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)
```

This allows us to see the matching score directly on the frame.

---

# 21. Process a Video

The previous steps work for one frame.

For a video, repeat them for every frame:

```python
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    result = cv2.matchTemplate(
        gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_score, _, max_location = cv2.minMaxLoc(
        result
    )

    top_left = max_location

    bottom_right = (
        top_left[0] + template_w,
        top_left[1] + template_h
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (0, 255, 0),
        2
    )
```

The important idea:

```text
Read frame
    ↓
Process frame
    ↓
Read next frame
    ↓
Process frame
    ↓
...
```

---

# 22. Save the Processed Video

Create a codec:

```python
fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)
```

Create the writer:

```python
writer = cv2.VideoWriter(
    "outputs/result.mp4",
    fourcc,
    fps,
    (width, height)
)
```

Save every processed frame:

```python
writer.write(frame)
```

At the end:

```python
writer.release()
```

---

# 23. Release Resources

When finished:

```python
cap.release()
```

For the video writer:

```python
writer.release()
```

This is important because OpenCV keeps resources open while working with the video.

---

# 24. Complete Pipeline

```text
                    VIDEO
                      │
                      ▼
               Read first frame
                      │
                      ▼
              Select / crop template
                      │
                      ▼
                  TEMPLATE
                      │
                      ▼
               Convert to grayscale
                      │
                      ▼
              ┌─────────────────┐
              │   Every frame   │
              └────────┬────────┘
                       │
                       ▼
                Grayscale frame
                       │
                       ▼
                matchTemplate()
                       │
                       ▼
                 Result matrix
                       │
                       ▼
                  minMaxLoc()
                       │
                       ▼
                  Best location
                       │
                       ▼
                 Bounding box
                       │
                       ▼
                 Draw detection
                       │
                       ▼
                  Save frame
                       │
                       ▼
                  Next frame
                       │
                       └──────→ repeat
```

---

# 25. Minimal Version

Once the individual steps are understood, the complete algorithm becomes much smaller:

```python
import cv2

cap = cv2.VideoCapture("assets/cup.mp4")

template = ...

template_gray = cv2.cvtColor(
    template,
    cv2.COLOR_BGR2GRAY
)

template_h, template_w = template_gray.shape

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    result = cv2.matchTemplate(
        gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    _, score, _, location = cv2.minMaxLoc(
        result
    )

    top_left = location

    bottom_right = (
        top_left[0] + template_w,
        top_left[1] + template_h
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (0, 255, 0),
        2
    )

cap.release()
```

---

# 26. Limitations of Basic Template Matching

Template Matching is simple and useful, but it has limitations.

It can struggle when the object:

- changes scale
- rotates significantly
- changes appearance
- becomes partially hidden
- changes perspective
- becomes very small
- moves very fast

Example:

```text
Template
┌──────────┐
│   CUP    │
└──────────┘

        ↓ scale change

    ┌───────────────┐
    │      CUP      │
    └───────────────┘

Basic Template Matching may fail.
```

---

# 27. Next Step — Robust Template Matching

The next version improves the basic approach.

Instead of searching the entire frame every time:

```text
Entire Frame
┌───────────────────────────────┐
│                               │
│             OBJECT            │
│                               │
└───────────────────────────────┘
```

we can search around the object's previous position:

```text
Previous position
        ↓
      ROI
        ↓
Template Matching
        ↓
New position
        ↓
Update ROI
        ↓
Next frame
```

This can make the system more efficient and more stable.

---

# Quick Mental Model

If you remember only one thing:

```text
TEMPLATE
   +
CURRENT FRAME
   ↓
matchTemplate()
   ↓
SCORE MAP
   ↓
minMaxLoc()
   ↓
BEST LOCATION
   ↓
BOUNDING BOX
```

**That is the core idea behind Template Matching.**