import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class DeepfakeDetectorModel(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.5):
        super(DeepfakeDetectorModel, self).__init__()
        
        # Load pre-trained ResNeXt50_32x4d
        if pretrained:
            weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
            self.backbone = resnext50_32x4d(weights=weights)
        else:
            self.backbone = resnext50_32x4d(weights=None)
            
        # The output of ResNeXt before the FC layer is 2048-dimensional
        in_features = self.backbone.fc.in_features # This is 2048
        
        # Remove the original FC layer (we just want the features)
        self.backbone.fc = nn.Identity()
        
        # Binary Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        # Classification
        logits = self.classifier(features)
        return logits

def set_parameter_requires_grad(model, feature_extracting):
    """
    Utility to freeze or unfreeze the backbone for transfer learning.
    """
    if feature_extracting:
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        for param in model.backbone.parameters():
            param.requires_grad = True

def get_model(pretrained=True, freeze_backbone=True):
    model = DeepfakeDetectorModel(pretrained=pretrained)
    
    # Phase 1: Freeze backbone, train only the head
    if freeze_backbone:
        set_parameter_requires_grad(model, feature_extracting=True)
        # Ensure classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    return model
