# EAS 510 Assignment 1: Digital Forensics Apprentice

This project implements a rule-based expert system that matches modified images back to their originals using various computer vision techniques.

## Setup & Workflow

### 1. Installation
Install the necessary dependencies to run the system:
```bash
pip install -r requirements.txt
```

### 2. Running the System
To evaluate all images across the Phase 1 ("easy" & "random") and Phase 2 ("hard") datasets, simply execute the main test runner:
```bash
python test_system.py
```
This will generate `results_v1.txt`, `results_v1_hard.txt`, and `results_v2.txt` containing the detailed scoring outputs for every evaluated image.

### 3. Evaluating Accuracy
An extra utility script, `validating_result.py`, has been included to automatically compare the output text files against `ground_truth.json` and calculate the system's accuracy.
```bash
python validating_result.py
```

### 4. Project Structure
```text
.
├── forensics_detective.py  # Main SimpleDetective class implementation
├── rules.py                # Phase 1 rules (Rules 1-3)
├── rules_v2.py             # Phase 2 rules (Rule 4 ORB Keypoints)
├── test_system.py          # Main execution script to generate results
├── validating_result.py    # Custom evaluation script to check accuracy vs ground truth
├── ground_truth.json       # Labels mapping modified images to their original
├── requirements.txt        # Python dependency list
├── README.md               # Original assignment specification
├── MYREADME.md             # This documentation and phase reflections
│
├── originals/              # Folder: 10 original target images
├── modified_images/        # Folder: 60 easy transformations
├── hard/                   # Folder: 60 hard combined transformations
└── random/                 # Folder: 15 unrelated random images
```

---

## Part 1: The Detective (`rules.py`, `forensics_detective.py`)
This file is like a brain of the system in this file:
Firts it memorizes the original image and matches with the modified image 

Inorder to do so there are two steps:

## 1: This is the process to convert the actual picturs into number so that the computer can read it we used two methord here so that we can get accurate image number 

### First: ORB (Oriented FAST and Rotated BRIEF)

This methord is like taking a pen and putting a dot on every sharp corner or edge in a photo. These dots are called "Keypoints". this helps us to identiy fy all the shapes even if its modified 
For example even if you rotate the building the edge of the building is still present there and as its already noted by the system it can be easyly identiify 

**The Code**:
```python
# Find up to 4000 sharp corners/edges
self.orb = cv2.ORB_create(nfeatures=4000)
kp, des = self.orb.detectAndCompute(gray, None)
```

## Second: Color Histograms

This methord is helps us to identify the color we dont care abou the shape or anything like in ORB here we only deal with color So 
This methord is helps us to identify the color we dont care abou the shape or anything like in ORB here we only deal with color So if an image is extremely blurry, the "corners" disappear, but the colors usually stay the same.

**The Code**:
```python
# Count pixel colors in 3D space (Blue, Green, Red)
hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, ...])
```

## 2. The "Brain" (Decision Logic)
When the system analyzes an image, it follows a checklist (Rules).

#### Rule #1: The Geometry Check (Strongest)
It tries to match the Keypoints from the unknown image to the original.
1.  **KNN Match**: For every point in the unknown image,    find the 2 closest points in the original.
2.  **Ratio Test**: Throw away matches that look confusing.
3.  **RANSAC**: It tries to find a geometric pattern (like a rotation). It counts how many points fit this pattern. These are called **"Inliers"**.

 **Logic**: If `Inliers >= 18`, we are confident it's a match.

#### Rule #2: The Color Check (Fallback)
If the Geometry Check fails (maybe the image is too blurry), we check the color histogram.

 **Logic**: If `Color Correlation > 0.98` (98% similar), we say it is a match.

#### Rule #3: Rejection
If neither rule passes, we assume the image is random or unrelated.

## 3. Phase 1 Rules Summary

#### Rule 1: Metadata Analysis
- **What it does:** Compares the structural sizes of the unknown image to the target original image by comparing differences in file size (bytes) and image dimensions (width x height).
- **Why it works:** This is highly effective for detecting images that have been heavily compressed (lossy JPEG compression shrinks file sizes) or aggressively cropped (altering the width/height pixel dimensions).

#### Rule 2: Color Histogram
- **What it does:** Uses `cv2.calcHist()` to map the distribution of pixel colors across the BGR color space, then uses `cv2.compareHist()` with the correlation metric to see how similar the color distributions are.
- **Why it works:** BGR histograms are globally persistent. Even if an image is scaled, scrambled, or slightly cropped, the overall footprint of colors remains relatively stable. If the histogram correlation is very high, they are likely the same image.

#### Rule 3: Template Matching
- **What it does:** Resizes both images to have a max dimension of 400px for performance, converts them to grayscale, and uses `cv2.matchTemplate()` to slide the unknown image across the target image to find a localized match.
- **Why it works:** This is the most accurate rule for identifying direct crops, as it algorithmically proves that the exact pixel structure of the unknown image exists inside the original image.

---

## Part 2: Phase 2 Iteration and Reflection

### 1. Observed Weakness in V1 (Phase 1)
When testing the Phase 1 system against the `hard` dataset, accuracy dropped to **83.3%**. The system systematically struggled with **severe off-center crops**, **rotations**, and **contrast adjustments**. 
- **Reasoning:** Rule 1 relies on consistent sizing, Rule 2 relies on consistent color lighting, and Rule 3 (Template Matching) is extremely brittle regarding angle changes. When an image was rotated even barely (e.g. 6 degrees), template matching failed completely to overlap the pixels, resulting in false rejections for almost all rotated images.

### 2. Design Decision for V2 (Rule 4)
To address this, we implemented **Rule 4: ORB Keypoint Matching**.
- **Rationale:** ORB (Oriented FAST and Rotated BRIEF) is a feature extraction algorithm specifically designed to be robust against rotation, scaling, and contrast shifting. Rather than matching the global image, it finds high-contrast "interest points" (corners/edges) and matches those localized points geometrically via Brute-Force KNN matching and RANSAC. This allows it to identify a match even if only a small, rotated fragment is provided.

### 3. Effect of the Change
- **V1 Performance:** 83.3% accuracy on `hard` images.
- **V2 Performance:** Accuracy skyrocketed to **100%** on `hard` images (and **99.3%** system overall). All Previously failed hard cases involving rotation (`v4`) and off-center crops (`v1`) that bypassed Rule 3 now successfully fired Rule 4 with high confidence.

### 4. Trade-offs
- **Performance:** Rule 4 is highly computationally expensive. Extracting up to 4000 keypoints and cross-checking them via Brute-Force drastically increased the runtime per image compared to simple histogram checks.
- **Complexity:** Keypoint matching requires very careful parameter tuning (`distance < 0.75` ratio tests, minimum match counts) to avoid false-positive matches occurring inside repeating geometric textures (like brick walls or grass).