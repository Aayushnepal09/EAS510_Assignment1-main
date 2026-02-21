import cv2
import numpy as np
import os
from PIL import Image

def rule1(target_data, unknown_img_path):
    """
    Rule 1: Metadata Analysis
    Compare file size and dimensions.
    """
    t_size = target_data['file_size']
    t_w, t_h = target_data['dimensions']
    
    u_stat = os.stat(unknown_img_path)
    u_size = u_stat.st_size
    
    with Image.open(unknown_img_path) as u_img:
        u_w, u_h = u_img.size
        
    size_ratio = min(u_size, t_size) / max(u_size, t_size)
    dim_ratio = min(u_w * u_h, t_w * t_h) / max(u_w * u_h, t_w * t_h)
    
    final_score = int(size_ratio * 30)
    
    return final_score, f"Size ratio {size_ratio:.2f}"

def rule2(target_data, unknown_img):
    """
    Rule 2: Color Histogram Analysis
    Compare color distributions.
    """
    hist_u = cv2.calcHist([unknown_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_u, hist_u)
    
    hist_t = target_data['histogram']
    
    correlation = cv2.compareHist(hist_u, hist_t, cv2.HISTCMP_CORREL)
    points = int(max(0, correlation) * 30)
    
    return points, f"Correlation {correlation:.2f}"

def rule3(target_data, unknown_img):
    """
    Rule 3: Template Matching
    Optimized: Resize images to a smaller scale for faster processing.
    """
    MAX_DIM = 400
    
    target = target_data['image']
    template = unknown_img
    
    h_t, w_t = target.shape[:2]
    h_u, w_u = template.shape[:2]
    
    scale_t = MAX_DIM / max(h_t, w_t)
    target_small = cv2.resize(target, None, fx=scale_t, fy=scale_t)
    
    template_small = cv2.resize(template, None, fx=scale_t, fy=scale_t)
    
    h_ts, w_ts = target_small.shape[:2]
    h_us, w_us = template_small.shape[:2]
    
    if h_us > h_ts or w_us > w_ts:
        scale_u = min(h_ts/h_us, w_ts/w_us)
        template_small = cv2.resize(template_small, None, fx=scale_u, fy=scale_u)

    target_gray = cv2.cvtColor(target_small, cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(template_small, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(target_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    
    points = int(max(0, max_val) * 40)
    
    return points, f"Match score {max_val:.2f}"
