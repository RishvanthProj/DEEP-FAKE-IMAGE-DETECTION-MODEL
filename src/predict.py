import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from src.utils import get_device
from src.model import get_model
from src.preprocessing import process_image
from src.face_detection import FaceDetector
from src.explainability import generate_gradcam, overlay_heatmap

class DeepfakePredictor:
    def __init__(self, model_path="models/deepfake_resnext50_final.pth"):
        self.device = get_device()
        self.model = get_model(pretrained=False, freeze_backbone=False).to(self.device)
        
        # Default fallback
        self.use_face_crop = True
        
        # Load metadata if exists
        metadata_path = "models/model_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                import json
                meta = json.load(f)
                if 'use_face_crop' in meta:
                    self.use_face_crop = meta['use_face_crop']
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.is_ready = True
        else:
            self.is_ready = False
            
        self.face_detector = FaceDetector(device=self.device)
        
    def predict(self, image: Image.Image):
        if not self.is_ready:
            return {"error": "Model not trained or checkpoint not found."}
            
        start_time = time.time()
        
        original_image = image.convert('RGB')
        
        face_info = {
            "face_detected": False,
            "faces_count": 0,
            "bbox": None,
            "confidence": 0.0
        }
        
        # 1. ALWAYS run face detection for the gate
        all_boxes, all_probs = self.face_detector.detect_faces(original_image)
        
        # 2. Validate detections
        valid_faces = 0
        best_prob = 0
        best_box = None
        
        FACE_CONFIDENCE_THRESHOLD = 0.85
        MIN_AREA = 400 # e.g. 20x20
        
        if all_boxes:
            for box, prob in zip(all_boxes, all_probs):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                
                if prob >= FACE_CONFIDENCE_THRESHOLD and area >= MIN_AREA:
                    valid_faces += 1
                    if prob > best_prob:
                        best_prob = prob
                        best_box = box
                        
        face_info["faces_count"] = valid_faces
        face_info["confidence"] = float(best_prob) if best_box is not None else 0.0
        face_info["bbox"] = best_box
        
        if valid_faces == 0:
            # STOP ANALYSIS! No human face.
            face_info["face_detected"] = False
            return {
                "prediction": "NO_FACE",
                "face_info": face_info,
                "analysis_image": original_image,
                "image_size": original_image.size
            }
            
        # Face is found!
        face_info["face_detected"] = True
        
        analysis_image = original_image
        img_w, img_h = original_image.size
        
        # If the image is a full photograph or selfie (larger than standard dataset crops ~300x300),
        # extract the head/face region with natural margin matching dataset proportions
        if best_box is not None and (img_w > 300 or img_h > 300):
            bw = best_box[2] - best_box[0]
            bh = best_box[3] - best_box[1]
            if (bw / img_w) < 0.65 or (bh / img_h) < 0.65:
                cx = (best_box[0] + best_box[2]) / 2.0
                cy = (best_box[1] + best_box[3]) / 2.0
                crop_size = max(bw, bh) / 0.45
                half = crop_size / 2.0
                x1 = max(0, int(cx - half))
                y1 = max(0, int(cy - half))
                x2 = min(img_w, int(cx + half))
                y2 = min(img_h, int(cy + half))
                if x2 > x1 and y2 > y1:
                    analysis_image = original_image.crop((x1, y1, x2, y2))
                
        # Preprocess
        input_tensor = process_image(analysis_image, transform_type='predict').to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
        prob_real = float(probs[0])
        prob_fake = float(probs[1])
        
        # Identify prediction
        predicted_class = "DEEPFAKE" if prob_fake > 0.5 else "REAL"
        confidence = prob_fake if predicted_class == "DEEPFAKE" else prob_real
        
        # Grad-CAM
        try:
            target_cls = 1 if predicted_class == "DEEPFAKE" else 0
            heatmap = generate_gradcam(self.model, input_tensor, target_class=target_cls)
            
            # Prepare image for overlay
            resized_img = analysis_image.resize((224, 224))
            img_np = np.array(resized_img)
            cam_overlay = overlay_heatmap(img_np, heatmap)
            
            cam_image = Image.fromarray(cam_overlay)
        except Exception as e:
            cam_image = None
            print(f"Grad-CAM error: {e}")
            
        inference_time = (time.time() - start_time) * 1000 # ms
        
        # Uncertainty
        uncertain = False
        if 0.45 <= confidence <= 0.55:
            uncertain = True
            
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "uncertain": uncertain,
            "face_info": face_info,
            "inference_time_ms": inference_time,
            "analysis_image": analysis_image,
            "gradcam_image": cam_image,
            "image_size": original_image.size
        }
