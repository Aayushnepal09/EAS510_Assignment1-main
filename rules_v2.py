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
    
    gray_u = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2GRAY)
    kp_u, des_u = orb.detectAndCompute(gray_u, None)
    
    gray_t = cv2.cvtColor(target_data['image'], cv2.COLOR_BGR2GRAY)
    kp_t, des_t = orb.detectAndCompute(gray_t, None)
    
    inliers_count = 0
    if des_u is not None and des_t is not None and len(des_u) >= 2 and len(des_t) >= 2:
        knn = bf.knnMatch(des_u, des_t, k=2)
        good = []
        for pair in knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        if len(good) >= 12:
            src_pts = np.float32([kp_u[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers_count = int(mask.sum())
                
    points = int(min(50, inliers_count * 2)) 
    
    return points, f"ORB inliers {inliers_count}"
