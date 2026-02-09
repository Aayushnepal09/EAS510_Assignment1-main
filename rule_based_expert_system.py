import cv2
import numpy as np
import os
import glob

class ForensicExpertSystem:
    def __init__(self, originals_path):
       
        self.originals_path = originals_path
        self.knowledge_base = {} 
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)

        # Matcher 
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Load the database
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        
        print(f"Loading Knowledge Base from: {self.originals_path}")
        image_files = glob.glob(os.path.join(self.originals_path, "*"))
        
        for file_path in image_files:
            filename = os.path.basename(file_path)
            # color image 
            img = cv2.imread(file_path)
            
            if img is None:
                continue
                
            # for crop and rotration 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            
            # for brightness 
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)

            # adding value co compare with unknown image
            self.knowledge_base[filename] = {
                'keypoints': kp,
                'descriptors': des,
                'histogram': hist,
                'shape': img.shape
            }
            
        print(f"Knowledge Base built: {len(self.knowledge_base)} originals registered.\n")

    def analyze_image(self, unknown_image_path):
        
        unknown_img = cv2.imread(unknown_image_path)
        if unknown_img is None:
            return None, "Error: Could not read image"

        # Pre-process unknown image
        gray = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2GRAY)
        kp_u, des_u = self.orb.detectAndCompute(gray, None)
        
        hist_u = cv2.calcHist([unknown_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_u, hist_u)

        best_feature_score = 0
        best_feature_match = None
        
        best_hist_score = -1.0   # FIX: correlation can be negative
        best_hist_match = None

        # Compare against every original in the database
        for original_name, data in self.knowledge_base.items():
            
            # --- RULE 1: Feature Matching

            inliers_count = 0
        
            des_o = data['descriptors']
            kp_o = data['keypoints']

            if des_u is not None and des_o is not None and len(des_u) >= 2 and len(des_o) >= 2:
                
                # KNN match
                knn = self.bf.knnMatch(des_u, des_o, k=2)

                # Lowe ratio test (filters weak/ambiguous matches)
                good = []
                for pair in knn:
                    if len(pair) != 2:
                        continue
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                # If enough good matches, verify geometry using RANSAC
                if len(good) >= 12:
                    src_pts = np.float32([kp_u[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_o[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers_count = int(mask.sum())

            # Feature score = number of RANSAC inliers
            if inliers_count > best_feature_score:
                best_feature_score = inliers_count
                best_feature_match = original_name

            # --- RULE 2: Color Histogram Correlation ---
            hist_score = cv2.compareHist(hist_u, data['histogram'], cv2.HISTCMP_CORREL)
            if hist_score > best_hist_score:
                best_hist_score = hist_score
                best_hist_match = original_name

        # --- FINAL DECISION LOGIC ---
        
        # Rule A: Strong geometric match using inliers (handles crops/rotation reliably)
        if best_feature_score >= 18:
            return best_feature_match, f"MATCH FOUND: {best_feature_match} | Reasoning: Strong geometric match ({best_feature_score} RANSAC inliers)."
        
        # Rule B: Fallback to color histogram if features fail (e.g., heavy blur/compression)
        elif best_hist_score > 0.98:
            return best_hist_match, f"MATCH FOUND: {best_hist_match} | Reasoning: High color similarity ({best_hist_score:.3f}) fallback."
            
        # Default: Reject
        return "REJECTED", f"NO MATCH: Max geometric inliers: {best_feature_score}, Max color correlation: {best_hist_score:.2f}"
        
def main():
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Use the script's directory
    originals_path = os.path.join(base_dir, "originals")
    
    # Check if directories exist
    if not os.path.exists(originals_path):
        print("Error: Directory structure not found. Please check paths.")
        return

    # 2. Initialize System
    expert = ForensicExpertSystem(originals_path)

    # 3. Define test directories
    test_dirs = {
        "Easy Cases": os.path.join(base_dir, "modified_images"),
        "Hard Cases": os.path.join(base_dir, "hard"),
        "Random Cases": os.path.join(base_dir, "random")
    }

    # 4. Run Analysis
    for category, path in test_dirs.items():
        print(f"\n--- Analyzing: {category} ---")
        if not os.path.exists(path):
            print(f"Skipping {category} (folder not found)")
            continue

        images = glob.glob(os.path.join(path, "*"))
        images.sort()

        for img_path in images:
            filename = os.path.basename(img_path)
            
            # Ask the expert system
            result, reason = expert.analyze_image(img_path)
            
            # Print output (Tabular style for readability)
            status_icon = "✅" if result != "REJECTED" else "❌"
            
            
            print(f"{status_icon} File: {filename[:25]}... -> {result}")
            print(f"   └── {reason}")

if __name__ == "__main__":
    main()
