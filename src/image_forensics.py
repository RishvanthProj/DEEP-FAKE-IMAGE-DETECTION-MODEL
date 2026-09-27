import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def generate_ela(image_path, quality=90):
    """
    Error Level Analysis (ELA)
    Saves image at a specific quality, then finds the difference.
    """
    original = Image.open(image_path).convert('RGB')
    
    # Save temporarily to memory/disk
    import tempfile
    import os
    temp_filename = "temp_ela.jpg"
    original.save(temp_filename, 'JPEG', quality=quality)
    
    compressed = Image.open(temp_filename)
    ela_image = ImageChops.difference(original, compressed)
    
    # Enhance the difference
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return ela_image

def generate_edge_map(image_np):
    """
    Edge detection using Canny.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
        
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges)

def generate_noise_residual(image_np):
    """
    High-frequency noise residual (Image - MedianBlur).
    """
    # Median blur removes high frequency noise
    blurred = cv2.medianBlur(image_np, 5)
    # The difference highlights the noise
    residual = cv2.absdiff(image_np, blurred)
    
    # Amplify for visualization
    residual = cv2.convertScaleAbs(residual, alpha=2.0, beta=0)
    return Image.fromarray(residual)
