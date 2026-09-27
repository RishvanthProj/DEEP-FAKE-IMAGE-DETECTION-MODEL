import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.utils import set_seed, get_device, setup_logging, save_json
from src.dataset import discover_dataset
from src.preprocessing import get_train_transforms, get_eval_transforms, DeepfakeDataset
from src.model import get_model, set_parameter_requires_grad
from src.face_detection import FaceDetector

logger = setup_logging()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Detector Model")
    parser.add_argument("--train-dir", type=str, default="/Users/rishvantha/Downloads/Deepfake Dataset/Train", help="Train dataset")
    parser.add_argument("--val-dir", type=str, default="/Users/rishvantha/Downloads/Deepfake Dataset/Validation", help="Validation dataset")
    parser.add_argument("--test-dir", type=str, default="/Users/rishvantha/Downloads/Deepfake Dataset/Test", help="Test dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs-head", type=int, default=5, help="Epochs for training head")
    parser.add_argument("--epochs-finetune", type=int, default=10, help="Epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for head")
    parser.add_argument("--lr-finetune", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--use-face-crop", action="store_true", help="Use MTCNN face crop for training")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny smoke test")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split")
    
    args = parser.parse_args()
    
    set_seed(42)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Dataset discovery
    import os
    import pandas as pd
    # Always regenerate the manifest to ensure we are using the correct paths
    logger.info("Discovering dataset and regenerating manifest...")
    df = discover_dataset(args.train_dir, args.val_dir, args.test_dir)
        
    if df is None:
        logger.error("Failed to discover dataset.")
        return
        
    # Smoke test limits
    if args.smoke_test:
        logger.info("SMOKE TEST MODE: Limiting dataset to 32 samples per split")
        df = df.groupby('split').head(32).reset_index(drop=True)
        args.epochs_head = 1
        args.epochs_finetune = 1
        args.batch_size = 4
    elif args.max_samples:
        logger.info(f"Limiting dataset to {args.max_samples} samples per split")
        df = df.groupby('split').apply(lambda x: x.sample(n=min(len(x), args.max_samples), random_state=42)).reset_index(drop=True)
        
    train_df = df[df['split'] == 'Train']
    val_df = df[df['split'] == 'Validation']
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Calculate class weights for Imbalance
    train_labels = [0 if l == 'REAL' else 1 for l in train_df['label']]
    unique_classes = np.unique(train_labels)
    if len(unique_classes) > 1:
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    else:
        class_weights = [1.0, 1.0]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    logger.info(f"Class weights (REAL, FAKE): {class_weights}")
    
    face_detector = FaceDetector(device=device) if args.use_face_crop else None
    
    train_dataset = DeepfakeDataset(train_df, transform=get_train_transforms(), use_face_crop=args.use_face_crop, face_detector=face_detector)
    val_dataset = DeepfakeDataset(val_df, transform=get_eval_transforms(), use_face_crop=args.use_face_crop, face_detector=face_detector)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = get_model(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Phase 1: Train Head
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-3)
    logger.info("Phase 1: Training Classification Head")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(args.epochs_head):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs_head} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_head = epoch
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/deepfake_resnext50_final.pth")
            logger.info("Saved best model.")

    # Phase 2: Fine-Tuning
    logger.info("Phase 2: Fine-tuning entire model")
    set_parameter_requires_grad(model, feature_extracting=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_finetune, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    for epoch in range(args.epochs_finetune):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Fine-tune Epoch {epoch+1}/{args.epochs_finetune} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_finetune = epoch
            torch.save(model.state_dict(), "models/deepfake_resnext50_final.pth")
            logger.info("Saved best model.")

    # Save configs and metrics
    save_json(history, "results/training_history.json")
    
    test_df = df[df['split'] == 'Test']
    
    model_metadata = {
        "architecture": "ResNeXt50_32x4d",
        "class_mapping": {"REAL": 0, "FAKE": 1},
        "input_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "training_dataset_path": args.train_dir,
        "validation_dataset_path": args.val_dir,
        "test_dataset_path": args.test_dir,
        "training_timestamp": time.time(),
        "best_epoch": getattr(locals(), 'best_epoch_finetune', getattr(locals(), 'best_epoch_head', 0)),
        "num_train": len(train_df),
        "num_validation": len(val_df),
        "num_test": len(test_df),
        "seed": 42,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": 1e-3,
        "batch_size": args.batch_size,
        "use_face_crop": args.use_face_crop
    }
    save_json(model_metadata, "models/model_metadata.json")
    save_json({"0": "REAL", "1": "FAKE"}, "models/class_mapping.json")
    
    logger.info("Training complete. Running final evaluation on test set...")
    # Evaluation on Test set
    test_df = df[df['split'] == 'Test']
    if args.smoke_test:
        test_df = test_df.groupby('label').head(16).reset_index(drop=True)
    
    test_dataset = DeepfakeDataset(test_df, transform=get_eval_transforms(), use_face_crop=args.use_face_crop, face_detector=face_detector)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    from src.evaluate import evaluate_model
    evaluate_model(model, test_loader, device)
    
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
