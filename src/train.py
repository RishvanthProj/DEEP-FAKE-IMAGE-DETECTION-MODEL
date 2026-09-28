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
from src.model import get_model, set_parameter_requires_grad, set_fine_tune_layers
from src.face_detection import FaceDetector

logger = setup_logging()

def calibrate_model(model, val_loader, device):
    logger.info("Computing validation calibration and optimal operating threshold...")
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Temperature grid search (minimize NLL on validation set)
    best_nll = float('inf')
    best_t = 1.15
    for t in np.linspace(0.8, 2.0, 25):
        probs = F.softmax(all_logits / t, dim=1).numpy()
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        nll = -np.mean(np.log(probs[np.arange(len(all_labels)), all_labels]))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
            
    # Optimal operating threshold search on validation set
    from sklearn.metrics import f1_score
    cal_probs = F.softmax(all_logits / best_t, dim=1)[:, 1].numpy()
    best_f1 = 0.0
    best_th = 0.52
    for th in np.linspace(0.40, 0.65, 26):
        preds = (cal_probs >= th).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
            
    logger.info(f"Calibration Complete: Temperature={best_t:.3f}, Operating Threshold={best_th:.3f} (Val F1={best_f1:.4f})")
    return best_t, best_th

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
    manifest_file = "data/manifest.csv"
    if os.path.exists(manifest_file):
        logger.info(f"Loading cached dataset manifest from {manifest_file}...")
        df = pd.read_csv(manifest_file)
    else:
        logger.info("Discovering dataset and generating manifest...")
        df = discover_dataset(args.train_dir, args.val_dir, args.test_dir)
        
    if df is None or len(df) == 0:
        logger.error("Failed to discover dataset.")
        return
        
    # Balanced sampling per split and label
    if args.smoke_test:
        logger.info("SMOKE TEST MODE: Limiting dataset to 32 balanced samples per split")
        df = df.groupby(['split', 'label'], group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), 16), random_state=42)
        ).reset_index(drop=True)
        args.epochs_head = 1
        args.epochs_finetune = 1
        args.batch_size = 4
    elif args.max_samples:
        logger.info(f"Limiting dataset to balanced samples (max {args.max_samples} total per split)")
        train_per_class = max(1, args.max_samples // 2)
        eval_per_class = min(500, train_per_class)
        
        train_sub = df[df['split'] == 'Train'].groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), train_per_class), random_state=42)
        )
        val_sub = df[df['split'] == 'Validation'].groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), eval_per_class), random_state=42)
        )
        test_sub = df[df['split'] == 'Test'].groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), eval_per_class), random_state=42)
        )
        df = pd.concat([train_sub, val_sub, test_sub]).reset_index(drop=True)
        
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
    
    train_dataset = DeepfakeDataset(train_df, transform=get_train_transforms(), use_face_crop=args.use_face_crop, face_detector=face_detector, robust_augment_real=True)
    val_dataset = DeepfakeDataset(val_df, transform=get_eval_transforms(), use_face_crop=args.use_face_crop, face_detector=face_detector)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = get_model(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Phase 1: Train Head
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-3)
    logger.info("Phase 1: Training Classification Head with Smartphone Hard-Negative Augmentations")
    
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
            torch.save(model.state_dict(), "models/deepfake_resnext50_robust.pth")
            logger.info("Saved best model.")

    # Phase 2: Fine-Tuning Layer 4 and Head (Stable on Apple Silicon MPS)
    if args.epochs_finetune > 0:
        logger.info("Phase 2: Fine-tuning layer4 + classification head")
        set_fine_tune_layers(model, unfreeze_layer4_only=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_finetune, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
        
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
                torch.save(model.state_dict(), "models/deepfake_resnext50_robust.pth")
                logger.info("Saved best model.")

    # Save configs and metrics
    save_json(history, "results/training_history.json")
    
    # Run validation calibration
    val_temp, optimal_thresh = calibrate_model(model, val_loader, device)
    
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
        "best_epoch": locals().get('best_epoch_finetune', locals().get('best_epoch_head', 0)),
        "num_train": len(train_df),
        "num_validation": len(val_df),
        "num_test": len(test_df),
        "seed": 42,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": 1e-3,
        "batch_size": args.batch_size,
        "use_face_crop": args.use_face_crop,
        "temperature": val_temp,
        "optimal_threshold": optimal_thresh
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
