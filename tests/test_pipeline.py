import pytest
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from src.dataset import discover_dataset
from src.preprocessing import get_train_transforms, get_eval_transforms, process_image, DeepfakeDataset
from src.model import DeepfakeDetectorModel, get_model
from src.predict import DeepfakePredictor
from src.face_detection import FaceDetector

TRAIN_DIR = "/Users/rishvantha/Downloads/Deepfake Dataset/Train"
VAL_DIR = "/Users/rishvantha/Downloads/Deepfake Dataset/Validation"
TEST_DIR = "/Users/rishvantha/Downloads/Deepfake Dataset/Test"

def test_dataset_paths():
    assert os.path.exists(TRAIN_DIR), f"Train dir missing: {TRAIN_DIR}"
    assert os.path.exists(VAL_DIR), f"Val dir missing: {VAL_DIR}"
    assert os.path.exists(TEST_DIR), f"Test dir missing: {TEST_DIR}"

def test_class_mapping():
    # Real folder maps to REAL, Fake folder maps to FAKE
    df = discover_dataset(TRAIN_DIR, VAL_DIR, TEST_DIR, output_manifest="data/test_manifest.csv")
    assert df is not None
    real_paths = df[df['label'] == 'REAL']['filepath'].tolist()
    fake_paths = df[df['label'] == 'FAKE']['filepath'].tolist()
    
    for path in real_paths[:5]:
        assert 'Real' in path or 'real' in path
    for path in fake_paths[:5]:
        assert 'Fake' in path or 'fake' in path
        
def test_checkpoint_loads():
    model_path = "models/deepfake_resnext50_final.pth"
    if os.path.exists(model_path):
        predictor = DeepfakePredictor(model_path=model_path)
        assert predictor.is_ready
    else:
        pytest.skip("Model checkpoint not yet created.")

def test_model_outputs_two_logits():
    model = get_model(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2)

def test_softmax_sums_to_one():
    logits = torch.tensor([[1.5, -0.5], [0.0, 0.0]])
    probs = F.softmax(logits, dim=1)
    sums = probs.sum(dim=1).detach().numpy()
    assert np.allclose(sums, [1.0, 1.0])

def test_rgb_preprocessing():
    # create a dummy grayscale image
    img_gray = Image.new('L', (300, 300), color=128)
    tensor = process_image(img_gray, transform_type='predict')
    # Should be 3 channels
    assert tensor.shape[1] == 3

def test_image_size_consistent():
    img = Image.new('RGB', (500, 400), color=(128, 128, 128))
    train_tensor = process_image(img, transform_type='train')
    predict_tensor = process_image(img, transform_type='predict')
    assert train_tensor.shape == (1, 3, 224, 224)
    assert predict_tensor.shape == (1, 3, 224, 224)

def test_normalization_consistent():
    # Mean and Std should be applied the same
    # We can check the values if we feed a constant image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    t1 = process_image(img, transform_type='eval')
    t2 = process_image(img, transform_type='predict')
    assert torch.allclose(t1, t2)

def test_no_face_handling():
    detector = FaceDetector(device='cpu')
    # Image with no face (just noise)
    img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    face_crop, bbox = detector.get_primary_face_crop(img)
    assert face_crop is None
    assert bbox is None

def test_golden_images():
    model_path = "models/deepfake_resnext50_final.pth"
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet.")
        
    predictor = DeepfakePredictor(model_path=model_path)
    df = discover_dataset(TRAIN_DIR, VAL_DIR, TEST_DIR, output_manifest="data/test_manifest.csv")
    
    real_paths = df[(df['label'] == 'REAL') & (df['split'] == 'Train')]['filepath'].tolist()
    fake_paths = df[(df['label'] == 'FAKE') & (df['split'] == 'Train')]['filepath'].tolist()
    
    if len(real_paths) > 0:
        img_real = Image.open(real_paths[0])
        res_real = predictor.predict(img_real)
        assert res_real['prediction'] == 'REAL', f"Golden Real failed, got {res_real['prediction']}"
        
    if len(fake_paths) > 0:
        img_fake = Image.open(fake_paths[0])
        res_fake = predictor.predict(img_fake)
        assert res_fake['prediction'] == 'DEEPFAKE', f"Golden Fake failed, got {res_fake['prediction']}"
