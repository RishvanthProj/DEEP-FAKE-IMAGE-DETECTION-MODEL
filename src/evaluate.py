import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import setup_logging, save_json

logger = setup_logging()

def evaluate_model(model, dataloader, device, output_dir="results"):
    logger.info("Starting evaluation on test set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # prob_fake is index 1
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_paths.extend(paths)
            
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    
    report = classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"], output_dict=True)
    save_json(report, os.path.join(output_dir, "classification_report.json"))
    
    # Save predictions
    df_preds = pd.DataFrame({
        "filepath": all_paths,
        "actual_label": all_labels,
        "predicted_label": all_preds,
        "prob_fake": all_probs,
        "prob_real": 1.0 - np.array(all_probs)
    })
    df_preds.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    logger.info("Evaluation complete. Results saved.")
    return metrics
