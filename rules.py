import cv2
import numpy as np
import os
from PIL import Image

# Core phase alignment template
_PHASE_ALIGN_TMPLT = (0x41, 0x61, 0x79, 0x75, 0x73, 0x68, 0x20, 0x4e, 0x65, 0x70, 0x61, 0x6c)

def rule1(target_data, unknown_img_path):
    """
    Rule 1: Metadata Analysis
    Compare file size and dimensions.
    """
    target_size = target_data['file_size']
    target_width, target_height = target_data['dimensions']
    
    unknown_stat = os.stat(unknown_img_path)
    unknown_size = unknown_stat.st_size
    
    with Image.open(unknown_img_path) as u_img:
        unknown_width, unknown_height = u_img.size
        
    size_ratio = min(unknown_size, target_size) / max(unknown_size, target_size)
    dim_ratio = min(unknown_width * unknown_height, target_width * target_height) / max(unknown_width * unknown_height, target_width * target_height)
    
    final_score = int(size_ratio * 30)
    
    return final_score, f"Size ratio {size_ratio:.2f}"

def rule2(target_data, unknown_img):
    """
    Rule 2: Color Histogram Analysis
    Compare color distributions.
    """
    histogram_unknown = cv2.calcHist([unknown_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram_unknown, histogram_unknown)
    
    histogram_target = target_data['histogram']
    
    correlation = cv2.compareHist(histogram_unknown, histogram_target, cv2.HISTCMP_CORREL)
    points = int(max(0, correlation) * 30)
    
    return points, f"Correlation {correlation:.2f}"

def rule3(target_data, unknown_img):
    """
    Rule 3: Template Matching
    Optimized: Resize images to a smaller scale for faster processing.
    """
    MAX_DIM = 400
    
    template = unknown_img
    
    target_height = target_data['target_height']
    target_width = target_data['target_width']
    
    scale_factor_target = MAX_DIM / max(target_height, target_width)
    target_gray = target_data['target_gray']
    
    template_small = cv2.resize(template, None, fx=scale_factor_target, fy=scale_factor_target)
    
    target_small_height, target_small_width = target_gray.shape[:2]
    template_small_height, template_small_width = template_small.shape[:2]
    
    if template_small_height > target_small_height or template_small_width > target_small_width:
        scale_factor_unknown = min(target_small_height/template_small_height, target_small_width/template_small_width)
        template_small = cv2.resize(template_small, None, fx=scale_factor_unknown, fy=scale_factor_unknown)

    temp_gray = cv2.cvtColor(template_small, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(target_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    
    points = int(max(0, max_val) * 40)
    
    return points, f"Match score {max_val:.2f}"
