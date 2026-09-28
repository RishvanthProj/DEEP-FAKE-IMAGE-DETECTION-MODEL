import os
import time
import json
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from src.utils import get_device
from src.model import get_model
from src.preprocessing import process_image, get_tta_batch
from src.face_detection import FaceDetector
from src.explainability import generate_gradcam, overlay_heatmap
from src.gemini_analyzer import GeminiForensicAnalyzer
from src.decision_fusion import fuse_forensic_decisions

class DeepfakePredictor:
    def __init__(self, model_path="models/deepfake_resnext50_final.pth", gemini_api_key=None):
        self.device = get_device()
        self.model = get_model(pretrained=False, freeze_backbone=False).to(self.device)
        
        # Default fallback configurations
        self.use_face_crop = True
        self.temperature = 1.15
        self.threshold_fake = 0.55
        self.threshold_real = 0.45
        
        # Load metadata if exists
        metadata_path = "models/model_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                try:
                    meta = json.load(f)
                    if 'use_face_crop' in meta:
                        self.use_face_crop = meta['use_face_crop']
                    if 'temperature' in meta:
                        self.temperature = float(meta['temperature'])
                    if 'optimal_threshold' in meta:
                        self.threshold_fake = float(meta['optimal_threshold'])
                        self.threshold_real = 1.0 - self.threshold_fake
                except Exception as e:
                    print(f"Error loading metadata: {e}")
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.model.eval()
            self.is_ready = True
        else:
            self.is_ready = False
            
        self.face_detector = FaceDetector(device=self.device)
        self.gemini_analyzer = GeminiForensicAnalyzer(api_key=gemini_api_key)
        
    def predict(self, image: Image.Image, run_gemini: bool = True, use_tta: bool = True):
        if not self.is_ready:
            return {"error": "Model not trained or checkpoint not found."}
            
        start_time = time.time()
        original_image = image.convert('RGB')
        img_w, img_h = original_image.size
        
        face_info = {
            "face_detected": False,
            "faces_count": 0,
            "bbox": None,
            "confidence": 0.0,
            "face_quality": 0.0
        }
        
        # 1. Face Detection Validation Gate
        all_boxes, all_probs = self.face_detector.detect_faces(original_image)
        
        valid_faces = 0
        best_prob = 0.0
        best_box = None
        
        FACE_CONFIDENCE_THRESHOLD = 0.80
        MIN_AREA = 350 # Minimum detectable face region
        
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
            face_info["face_detected"] = False
            fusion = fuse_forensic_decisions(
                ml_prediction="NO_FACE",
                ml_prob_real=0.5,
                ml_prob_fake=0.5,
                ml_calibrated_prob_fake=0.5,
                tta_mean_fake=0.5,
                tta_std_fake=0.0,
                tta_stability="HIGH",
                face_info=face_info,
                gemini_result=None
            )
            return {
                "prediction": "NO_FACE",
                "confidence": 0.0,
                "prob_real": 0.5,
                "prob_fake": 0.5,
                "face_info": face_info,
                "analysis_image": original_image,
                "image_size": original_image.size,
                "gradcam_image": None,
                "inference_time_ms": (time.time() - start_time) * 1000,
                "fusion": fusion
            }
            
        face_info["face_detected"] = True
        
        # Adaptive Framing: Extract head region with natural margin for large phone/camera photos
        analysis_image = original_image
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
                    
        # Face Quality Score (sharpness + MTCNN confidence)
        try:
            gray_crop = np.array(analysis_image.convert('L'))
            lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
            quality_score = min(1.0, (lap_var / 120.0) * best_prob)
            face_info["face_quality"] = float(round(quality_score, 3))
        except Exception:
            face_info["face_quality"] = 0.70
            
        # 2. ResNeXt-50 Primary Forward Pass
        input_tensor = process_image(analysis_image, transform_type='predict').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            raw_probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Temperature scaled calibrated probabilities
            calibrated_outputs = outputs / self.temperature
            calibrated_probs = F.softmax(calibrated_outputs, dim=1).cpu().numpy()[0]
            
        prob_real_raw = float(raw_probs[0])
        prob_fake_raw = float(raw_probs[1])
        prob_fake_calibrated = float(calibrated_probs[1])
        prob_real_calibrated = float(calibrated_probs[0])
        
        # 3. Test-Time Augmentation (TTA) Stability
        if use_tta:
            try:
                tta_batch = get_tta_batch(analysis_image, device=self.device)
                with torch.no_grad():
                    tta_outputs = self.model(tta_batch)
                    tta_probs = F.softmax(tta_outputs / self.temperature, dim=1)[:, 1].cpu().numpy()
                tta_mean = float(np.mean(tta_probs))
                tta_std = float(np.std(tta_probs))
                if tta_std < 0.07:
                    tta_stability = "HIGH"
                elif tta_std < 0.14:
                    tta_stability = "MEDIUM"
                else:
                    tta_stability = "LOW"
            except Exception as e:
                tta_mean = prob_fake_calibrated
                tta_std = 0.0
                tta_stability = "MEDIUM"
        else:
            tta_mean = prob_fake_calibrated
            tta_std = 0.0
            tta_stability = "HIGH"
            
        # Primary ML Decision
        predicted_class = "DEEPFAKE" if prob_fake_calibrated >= self.threshold_fake else "REAL"
        confidence = prob_fake_calibrated if predicted_class == "DEEPFAKE" else prob_real_calibrated
        
        # 4. Grad-CAM++ Activation Visualization
        try:
            target_cls = 1 if predicted_class == "DEEPFAKE" else 0
            heatmap = generate_gradcam(self.model, input_tensor, target_class=target_cls)
            resized_img = analysis_image.resize((224, 224))
            img_np = np.array(resized_img)
            cam_overlay = overlay_heatmap(img_np, heatmap)
            cam_image = Image.fromarray(cam_overlay)
        except Exception as e:
            cam_image = None
            
        # 5. Gemini Auxiliary Second-Opinion (if requested & configured)
        gemini_result = None
        if run_gemini and self.gemini_analyzer.is_configured:
            gemini_result = self.gemini_analyzer.analyze_image(original_image)
            
        # 6. Decision Fusion Layer
        fusion_decision = fuse_forensic_decisions(
            ml_prediction=predicted_class,
            ml_prob_real=prob_real_calibrated,
            ml_prob_fake=prob_fake_calibrated,
            ml_calibrated_prob_fake=prob_fake_calibrated,
            tta_mean_fake=tta_mean,
            tta_std_fake=tta_std,
            tta_stability=tta_stability,
            face_info=face_info,
            gemini_result=gemini_result,
            threshold_fake=self.threshold_fake,
            threshold_real=self.threshold_real
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            # Legacy keys preserved for backward compatibility
            "prediction": predicted_class,
            "confidence": confidence,
            "prob_real": prob_real_calibrated,
            "prob_fake": prob_fake_calibrated,
            "prob_real_raw": prob_real_raw,
            "prob_fake_raw": prob_fake_raw,
            "uncertain": not fusion_decision["is_certain"],
            "face_info": face_info,
            "inference_time_ms": inference_time,
            "analysis_image": analysis_image,
            "gradcam_image": cam_image,
            "image_size": original_image.size,
            
            # Enhanced forensic metadata
            "tta": {
                "mean": tta_mean,
                "std": tta_std,
                "mean_fake": tta_mean,
                "std_fake": tta_std,
                "stability": tta_stability
            },
            "gemini": gemini_result,
            "fusion": fusion_decision
        }
