# Template Matching

A hands-on Computer Vision practice for understanding template matching,
object tracking, motion estimation, and the limitations of simple tracking.

## Structure

```text
template-matching/
│
├── assets/
│   └── cup.mp4
│
├── outputs/
│   ├── cup_template_matching.mp4
│   └── cup_robust_template_matching.mp4
│
├── template_matching.py
├── robust_template_matching.py
└── README.md
```

## 1. Simple Template Matching

```bash
python template_matching.py
```

The program:

1. Opens `assets/cup.mp4`
2. Lets you select the cup in the first frame.
3. Uses that crop as the template.
4. Searches the whole frame using `cv2.matchTemplate()`.
5. Finds the best match using `cv2.minMaxLoc()`.
6. Draws the bounding box and matching score.
7. Saves the result to `outputs/cup_template_matching.mp4`.

### Main idea

```text
First frame
    ↓
Select object
    ↓
Crop = template
    ↓
Search whole next frame
    ↓
Best match
    ↓
Bounding box
```

## 2. Robust Template Matching

```bash
python robust_template_matching.py
```

The robust version adds:

- Moving ROI
- Template update
- Confidence threshold
- Position tracking
- Visualization of the search region

### Main idea

```text
Template
   ↓
Search nearby ROI
   ↓
Best match
   ↓
New position
   ↓
Crop new template
   ↓
Move ROI
   ↓
Next frame
```

## Why is the robust version better?

A fixed template can become less useful when:

- the object changes slightly in appearance,
- lighting changes,
- the object/camera moves,
- the object appears slightly different in consecutive frames.

Updating the template allows the reference to follow the object's current appearance.

## Important limitation: template drift

Updating the template every frame is powerful, but dangerous.

If one frame is matched incorrectly:

```text
Wrong match
    ↓
Wrong new template
    ↓
Next match becomes worse
    ↓
Tracking drifts away
```

The confidence threshold reduces this risk, but does not eliminate it.

## What to experiment with

### Experiment A
Move the pen slowly.

### Experiment B
Move the camera while keeping the pen still.

### Experiment C
Move the pen quickly.

### Experiment D
Move the pen closer/farther from the camera.

### Experiment E
Change the lighting.

Observe when the matching score decreases and when the tracker fails.

## Key concepts

- Template = cropped reference image
- ROI = region where we search
- Template Matching = finding the most similar image patch
- Matching Score = how similar the patch is to the template
- Tracking = repeating the search frame-by-frame
- Motion Estimation = comparing object positions between frames
- Template Update = adapting the reference to the current frame
- Template Drift = error accumulation after a wrong match
- Parallax = different apparent motion caused by different object depths
