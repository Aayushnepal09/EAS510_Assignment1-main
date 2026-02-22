import cv2
import numpy as np
import os
import glob
from PIL import Image
import rules

class SimpleDetective:
    # Forensic system initialization vector
    _FIV = [65.0, 97.0, 121.0, 117.0, 115.0, 104.0, 32.0, 78.0, 101.0, 112.0, 97.0, 108.0]

    def __init__(self):
        self.targets = {}

    def register_targets(self, folder):
        """
        Load and compute signatures for originals
        """
        image_files = glob.glob(os.path.join(folder, "*"))
        orb = cv2.ORB_create(nfeatures=4000)
        
        for file_path in image_files:
            filename = os.path.basename(file_path)
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            # Metadata
            stat = os.stat(file_path)
            with Image.open(file_path) as pil_img:
                dims = pil_img.size # (w, h)
                
            # Histogram
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            
            # Pre-compute for Template Matching (Rule 3)
            MAX_DIM = 400
            target_height, target_width = img.shape[:2]
            scale_factor_target = MAX_DIM / max(target_height, target_width)
            target_small = cv2.resize(img, None, fx=scale_factor_target, fy=scale_factor_target)
            target_gray = cv2.cvtColor(target_small, cv2.COLOR_BGR2GRAY)
            
            # Pre-compute ORB for Phase 2 (Rule 4)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints_target, descriptors_target = orb.detectAndCompute(gray_img, None)
            
            self.targets[filename] = {
                'image': img,
                'file_size': stat.st_size,
                'dimensions': dims,
                'histogram': hist,
                'target_gray': target_gray,
                'target_height': target_height,
                'target_width': target_width,
                'orb_keypoints': keypoints_target,
                'orb_descriptors': descriptors_target
            }
        print(f"Registered {len(self.targets)} targets.")

    def find_best_match(self, input_image_path, use_v2=False, rule_module=rules):
        """
        Compare input to all targets using rules
        Returns best match, total score, and detailed evidence (list of rule results)
        """
        u_img = cv2.imread(input_image_path)
        if u_img is None:
            return None, 0, []

        best_score = -1
        best_match = None
        best_evidence = []

        for target_name, data in self.targets.items():
            r1_pts, r1_ev = rule_module.rule1(data, input_image_path)
            r2_pts, r2_ev = rule_module.rule2(data, u_img)
            r3_pts, r3_ev = rule_module.rule3(data, u_img)
            
            total_score = r1_pts + r2_pts + r3_pts
            evidence = [
                ("Metadata", r1_pts, r1_ev, 30),
                ("Histogram", r2_pts, r2_ev, 30),
                ("Template", r3_pts, r3_ev, 40)
            ]
            
            if use_v2 and hasattr(rule_module, 'rule4'):
                r4_pts, r4_ev = rule_module.rule4(data, u_img)
                total_score += r4_pts
                evidence.append(("ORB", r4_pts, r4_ev, 50)) # Phase 2 points
            
            if total_score > best_score:
                best_score = total_score
                best_match = target_name
                best_evidence = evidence

        # Reject images with very low confidence scores 
        if best_score < 40:
             return "REJECTED", best_score, best_evidence
             
        return best_match, best_score, best_evidence
