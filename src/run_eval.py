import argparse
import torch
from torch.utils.data import DataLoader
from src.utils import get_device, setup_logging
from src.dataset import discover_dataset
from src.preprocessing import get_eval_transforms, DeepfakeDataset
from src.model import get_model
from src.evaluate import evaluate_model
import os
import pandas as pd

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = get_device()
    logger.info("Running evaluation on test set...")
    
    if os.path.exists("data/manifest.csv"):
        df = pd.read_csv("data/manifest.csv")
    else:
        df = discover_dataset("data")
        
    test_df = df[df['split'] == 'Test']
    
    if args.smoke_test:
        test_df = test_df.groupby('label').head(16).reset_index(drop=True)
        
    test_dataset = DeepfakeDataset(test_df, transform=get_eval_transforms(), use_face_crop=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = get_model(pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
    model.to(device)
    
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
