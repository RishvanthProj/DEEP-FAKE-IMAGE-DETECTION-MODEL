import pytest
import numpy as np
import torch
import torch.nn.functional as F

def test_probability_normalization():
    # Test that softmax outputs probabilities that sum to 1
    logits = torch.tensor([[2.5, 0.5]])
    probs = F.softmax(logits, dim=1)
    
    sum_probs = probs.sum().item()
    assert np.isclose(sum_probs, 1.0)
    
    prob_real = probs[0, 0].item()
    prob_fake = probs[0, 1].item()
    assert prob_real > 0 and prob_fake > 0
    assert prob_real > prob_fake
