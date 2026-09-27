import pytest
import os
import torch
import numpy as np
from PIL import Image
from src.predict import DeepfakePredictor

def generate_noise_image(size=(300, 300)):
    return Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))

def test_no_face_gate_rejects_noise():
    # If model_path doesn't exist, we can't fully test predict without mocking
    model_path = "models/deepfake_resnext50_final.pth"
    if not os.path.exists(model_path):
        pytest.skip("Model checkpoint not yet created.")
        
    predictor = DeepfakePredictor(model_path=model_path)
    
    # Random noise (no face)
    img_no_face = generate_noise_image()
    res = predictor.predict(img_no_face)
    
    assert res["prediction"] == "NO_FACE"
    assert res["face_info"]["face_detected"] is False
    assert res["face_info"]["faces_count"] == 0

def test_no_face_gate_critical_regression(monkeypatch):
    """
    CRITICAL REGRESSION TEST
    Mock model.__call__ to ensure it's NEVER called if no face is detected.
    """
    model_path = "models/deepfake_resnext50_final.pth"
    if not os.path.exists(model_path):
        pytest.skip("Model checkpoint not yet created.")
        
    predictor = DeepfakePredictor(model_path=model_path)
    
    # Flag to track if model was called
    model_called = False
    
    # Original forward pass
    original_forward = predictor.model.forward
    
    def mock_forward(*args, **kwargs):
        nonlocal model_called
        model_called = True
        return original_forward(*args, **kwargs)
        
    # Monkeypatch the model's forward
    monkeypatch.setattr(predictor.model, "forward", mock_forward)
    
    # Blank image (black)
    img_blank = Image.new('RGB', (300, 300), color=(0, 0, 0))
    res = predictor.predict(img_blank)
    
    assert res["prediction"] == "NO_FACE"
    assert model_called is False, "CRITICAL FAILURE: Model was called on a non-face image!"

def test_face_detection_gate_passes_face(monkeypatch):
    """
    Test that the gate allows processing if a face IS found.
    We mock detect_faces to return a high-confidence face.
    """
    model_path = "models/deepfake_resnext50_final.pth"
    if not os.path.exists(model_path):
        pytest.skip("Model checkpoint not yet created.")
        
    predictor = DeepfakePredictor(model_path=model_path)
    
    # Mock face detector to simulate finding a face
    def mock_detect_faces(img):
        return [[50, 50, 150, 150]], [0.99]
        
    def mock_get_primary_face_crop(img):
        return Image.new('RGB', (100, 100)), [50, 50, 150, 150]
        
    monkeypatch.setattr(predictor.face_detector, "detect_faces", mock_detect_faces)
    monkeypatch.setattr(predictor.face_detector, "get_primary_face_crop", mock_get_primary_face_crop)
    
    model_called = False
    original_forward = predictor.model.forward
    def mock_forward(*args, **kwargs):
        nonlocal model_called
        model_called = True
        return original_forward(*args, **kwargs)
        
    monkeypatch.setattr(predictor.model, "forward", mock_forward)
    
    img = Image.new('RGB', (300, 300), color=(128, 128, 128))
    res = predictor.predict(img)
    
    assert res["prediction"] != "NO_FACE"
    assert res["face_info"]["face_detected"] is True
    assert model_called is True
