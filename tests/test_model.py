import pytest
import torch
from src.model import DeepfakeDetectorModel, get_model

def test_model_initialization():
    model = get_model(pretrained=False)
    assert isinstance(model, DeepfakeDetectorModel)
    
def test_model_forward_pass():
    model = get_model(pretrained=False)
    model.eval()
    
    # Batch of 2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 2)
