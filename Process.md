## Part 1: The Detective (`rule_based_expert_system.py`)

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
1.  **KNN Match**: For every point in the unknown image, find the 2 closest points in the original.
2.  **Ratio Test**: Throw away matches that look confusing.
3.  **RANSAC**: It tries to find a geometric pattern (like a rotation). It counts how many points fit this pattern. These are called **"Inliers"**.

> **Logic**: If `Inliers >= 18`, we are confident it's a match.

#### Rule #2: The Color Check (Fallback)
If the Geometry Check fails (maybe the image is too blurry), we check the color histogram.

> **Logic**: If `Color Correlation > 0.98` (98% similar), we say it is a match.

#### Rule #3: Rejection
If neither rule passes, we assume the image is random or unrelated.

---

## Part 2: The Validation (`Validate_results.py`)

This script tests how good the system is.

### 1. The Correct Answers (`ground_truth.json`)
This file has the correct answer for every file.
*   `"original_01"` means it should match Original #1.
*   `null` means it should be REJECTED.

### 2. The Loop
The script does this for every file:
1.  **Run**: It runs `expert.analyze_image(file)`.
2.  **Compare**:
    *   System says: "It's Original 01".
    *   Correct Answer says: "It's Original 01".
    *   **Result**: Correct.
3.  **Record**: It saves the result to a list.

### 3. The Report
At the end, it calculates the score and saves two files:
1.  `predictions.json`: What the system said for every file.
2.  `comparison.json`: A detailed report card showing exactly which files passed or failed.

```python
accuracy = (correct_answers / total_questions) * 100
```