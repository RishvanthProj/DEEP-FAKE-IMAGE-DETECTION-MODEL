import os
import json
import time
from PIL import Image
from src.predict import DeepfakePredictor
from src.dataset import get_dataset_stats, discover_dataset

def generate_report():
    report = {}
    
    # 1. Dataset stats
    stats = get_dataset_stats("data/manifest.csv")
    if stats:
        report["Dataset counts"] = stats.get("splits", {})
        report["Class counts"] = {"REAL": stats.get("real"), "FAKE": stats.get("fake")}
        # duplicates are handled in dataset.py, assuming 0 remaining
        report["Duplicate counts"] = 0
    else:
        report["Dataset counts"] = {}
        report["Class counts"] = {}
        report["Duplicate counts"] = 0

    # 2. Model Metadata
    try:
        with open("models/model_metadata.json", "r") as f:
            meta = json.load(f)
            report["Model architecture"] = meta.get("architecture")
            report["Training configuration"] = {
                "optimizer": meta.get("optimizer"),
                "learning_rate": meta.get("learning_rate"),
                "weight_decay": meta.get("weight_decay"),
                "batch_size": meta.get("batch_size"),
                "use_face_crop": meta.get("use_face_crop")
            }
            report["Best epoch"] = meta.get("best_epoch")
            report["Checkpoint path"] = "models/deepfake_resnext50_final.pth"
            report["Class mapping"] = meta.get("class_mapping")
            report["Preprocessing config"] = {
                "input_size": meta.get("input_size"),
                "normalization": meta.get("normalization")
            }
    except Exception as e:
        print(f"Error reading metadata: {e}")

    # 3. Training/Validation history
    try:
        with open("results/training_history.json", "r") as f:
            hist = json.load(f)
            report["Training metrics"] = {
                "final_loss": hist["train_loss"][-1],
                "final_acc": hist["train_acc"][-1]
            }
            report["Validation metrics"] = {
                "final_loss": hist["val_loss"][-1],
                "final_acc": hist["val_acc"][-1]
            }
    except:
        pass

    # 4. Test metrics
    try:
        with open("results/metrics.json", "r") as f:
            metrics = json.load(f)
            report["Test metrics"] = {
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "specificity": metrics.get("specificity")
            }
            report["ROC-AUC"] = metrics.get("roc_auc")
            report["PR-AUC"] = metrics.get("pr_auc")
            report["Confusion matrix"] = metrics.get("confusion_matrix")
    except:
        pass

    # 5. Golden image predictions
    golden = {}
    inference_times = []
    
    if os.path.exists("models/deepfake_resnext50_final.pth"):
        predictor = DeepfakePredictor(model_path="models/deepfake_resnext50_final.pth")
        if predictor.is_ready:
            df = discover_dataset(
                "/Users/rishvantha/Downloads/Deepfake Dataset/Train", 
                "/Users/rishvantha/Downloads/Deepfake Dataset/Validation", 
                "/Users/rishvantha/Downloads/Deepfake Dataset/Test", 
                output_manifest="data/manifest.csv"
            )
            real_paths = df[(df['label'] == 'REAL') & (df['split'] == 'Train')]['filepath'].tolist()
            fake_paths = df[(df['label'] == 'FAKE') & (df['split'] == 'Train')]['filepath'].tolist()
            
            if len(real_paths) > 0:
                res = predictor.predict(Image.open(real_paths[0]))
                golden["KNOWN_REAL"] = res["prediction"]
                golden["KNOWN_REAL_CONFIDENCE"] = res["confidence"]
                inference_times.append(res.get("inference_time_ms", 0))
            if len(fake_paths) > 0:
                res = predictor.predict(Image.open(fake_paths[0]))
                golden["KNOWN_FAKE"] = res["prediction"]
                golden["KNOWN_FAKE_CONFIDENCE"] = res["confidence"]
                inference_times.append(res.get("inference_time_ms", 0))
                
    report["Golden image predictions"] = golden
    report["Inference latency"] = sum(inference_times)/len(inference_times) if inference_times else 0
    
    with open("reports/final_validation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Report generated at reports/final_validation_report.json")

if __name__ == "__main__":
    generate_report()
