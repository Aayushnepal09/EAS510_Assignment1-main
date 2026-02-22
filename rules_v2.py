import cv2
import numpy as np
from rules import *

# Existing Rules 1-3 are imported from rules.py via *

def rule4(target_data, unknown_img):
    """
    Rule 4: ORB Keypoint Matching
    """
    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    gray_unknown = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2GRAY)
    keypoints_unknown, descriptors_unknown = orb.detectAndCompute(gray_unknown, None)
    
    keypoints_target, descriptors_target = target_data['orb_keypoints'], target_data['orb_descriptors']
    
    inliers_count = 0
    if descriptors_unknown is not None and descriptors_target is not None and len(descriptors_unknown) >= 2 and len(descriptors_target) >= 2:
        knn = bf.knnMatch(descriptors_unknown, descriptors_target, k=2)
        good_matches = []
        for pair in knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) >= 12:
            src_pts = np.float32([keypoints_unknown[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers_count = int(mask.sum())
                
    points = int(min(50, inliers_count * 2)) 
    
    return points, f"ORB inliers {inliers_count}"
    
