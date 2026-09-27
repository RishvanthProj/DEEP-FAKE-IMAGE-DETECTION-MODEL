import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, input_tensor, target_class=1):
    """
    Generates Grad-CAM visualization for a given model and input tensor.
    target_class: 0 for REAL, 1 for FAKE
    """
    # ResNeXt final conv layer is usually layer4
    target_layers = [model.backbone.layer4[-1]]
    
    # We use GradCAM++ for better localization on single objects (faces)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(target_class)]
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

def overlay_heatmap(image_np, heatmap, alpha=0.5):
    """
    Overlay the Grad-CAM heatmap on the original image (numpy array RGB).
    """
    # Normalize image to [0,1] if not already
    if image_np.max() > 1.0:
        img = image_np.astype(np.float32) / 255.0
    else:
        img = image_np
        
    visualization = show_cam_on_image(img, heatmap, use_rgb=True, image_weight=alpha)
    return visualization
