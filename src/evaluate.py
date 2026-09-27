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
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(all_labels, all_probs)
    except Exception:
        pr_auc = 0.5
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
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
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor='#11161D')
    ax.set_facecolor('#0B0F14')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
                annot_kws={'size': 13, 'weight': 'bold', 'color': '#F3F4F6'}, ax=ax)
    ax.set_xlabel('Predicted Label', color='#9CA3AF', fontsize=11, labelpad=8)
    ax.set_ylabel('Actual Ground Truth', color='#9CA3AF', fontsize=11, labelpad=8)
    ax.set_title('Evaluation Confusion Matrix', color='#F3F4F6', fontsize=12, pad=12)
    ax.tick_params(colors='#9CA3AF')
    for spine in ax.spines.values():
        spine.set_color('#27303A')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close()

    # Plot ROC Curve
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#11161D')
        ax.set_facecolor('#0B0F14')
        ax.plot(fpr, tpr, color='#22C55E', lw=2.5, label=f'ResNeXt50 (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#6B7280', lw=1.5, linestyle='--', label='Baseline = 0.50')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', color='#9CA3AF', fontsize=11)
        ax.set_ylabel('True Positive Rate (Recall)', color='#9CA3AF', fontsize=11)
        ax.set_title('Receiver Operating Characteristic (ROC)', color='#F3F4F6', fontsize=12, pad=12)
        ax.tick_params(colors='#9CA3AF')
        for spine in ax.spines.values():
            spine.set_color('#27303A')
        ax.grid(color='#27303A', linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', facecolor='#11161D', edgecolor='#27303A', labelcolor='#F3F4F6')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=200, facecolor=fig.get_facecolor())
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot ROC curve: {e}")

    # Plot Precision-Recall Curve
    try:
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(all_labels, all_probs)
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#11161D')
        ax.set_facecolor('#0B0F14')
        ax.plot(rec, prec, color='#3B82F6', lw=2.5, label=f'ResNeXt50 (PR-AUC = {pr_auc:.3f})')
        ax.axhline(y=0.5, color='#6B7280', lw=1.5, linestyle='--', label='Baseline = 0.50')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', color='#9CA3AF', fontsize=11)
        ax.set_ylabel('Precision', color='#9CA3AF', fontsize=11)
        ax.set_title('Precision-Recall Curve', color='#F3F4F6', fontsize=12, pad=12)
        ax.tick_params(colors='#9CA3AF')
        for spine in ax.spines.values():
            spine.set_color('#27303A')
        ax.grid(color='#27303A', linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', facecolor='#11161D', edgecolor='#27303A', labelcolor='#F3F4F6')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=200, facecolor=fig.get_facecolor())
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot Precision-Recall curve: {e}")

    logger.info("Evaluation complete. Results saved.")
    return metrics
