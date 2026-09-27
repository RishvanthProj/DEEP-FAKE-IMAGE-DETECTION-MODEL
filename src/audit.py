import os
import glob
import json
import hashlib
from PIL import Image
import torch
import numpy as np
import pandas as pd

DATASET_ROOT = "/Users/rishvantha/Downloads/Deepfake Dataset"

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def audit_dataset():
    print("=" * 60)
    print("1. FULL DATASET AUDIT")
    print("=" * 60)
    
    splits = ["Train", "Validation", "Test"]
    classes = ["Real", "Fake"]
    
    counts = {}
    corrupt_count = 0
    formats = set()
    modes = set()
    sizes = set()
    
    for split in splits:
        counts[split] = {}
        for cls in classes:
            dir_path = os.path.join(DATASET_ROOT, split, cls)
            if not os.path.exists(dir_path):
                print(f"ERROR: Directory does not exist: {dir_path}")
                counts[split][cls] = 0
                continue
            
            files = [f for f in os.listdir(dir_path) if not f.startswith('.')]
            counts[split][cls] = len(files)
            
            # Check a sample of 50 images for corruptions and properties
            for f in files[:50]:
                fp = os.path.join(dir_path, f)
                try:
                    with Image.open(fp) as img:
                        formats.add(img.format)
                        modes.add(img.mode)
                        sizes.add(img.size)
                except Exception as e:
                    corrupt_count += 1
                    print(f"Corrupt image found: {fp} - {e}")
                    
    print("\nDataset Counts:")
    for split in splits:
        r = counts[split].get('Real', 0)
        f = counts[split].get('Fake', 0)
        total = r + f
        r_pct = (r / total * 100) if total > 0 else 0
        f_pct = (f / total * 100) if total > 0 else 0
        print(f"  {split:12s} Total: {total:7d} | Real: {r:6d} ({r_pct:.1f}%) | Fake: {f:6d} ({f_pct:.1f}%)")
        
    print(f"\nImage Properties Sampled:")
    print(f"  Formats: {formats}")
    print(f"  Color Modes: {modes}")
    print(f"  Sample Dimensions: {list(sizes)[:5]}")
    print(f"  Corrupt files in sample: {corrupt_count}")
    
    return counts

def audit_class_mapping():
    print("\n" + "=" * 60)
    print("2. CLASS MAPPING AUDIT")
    print("=" * 60)
    
    class_mapping = {
        0: "REAL",
        1: "FAKE"
    }
    
    # Save canonical class mapping
    os.makedirs("models", exist_ok=True)
    with open("models/class_mapping.json", "w") as f:
        json.dump({"0": "REAL", "1": "FAKE"}, f, indent=2)
    print("Saved canonical class mapping to models/class_mapping.json:")
    print(json.dumps(class_mapping, indent=2))
    
    return class_mapping

def audit_label_sanity():
    print("\n" + "=" * 60)
    print("3. LABEL SANITY TEST")
    print("=" * 60)
    
    class_to_idx = {"Real": 0, "Fake": 1, "REAL": 0, "FAKE": 1}
    idx_to_class = {0: "REAL", 1: "FAKE"}
    
    passes = True
    
    for cls in ["Real", "Fake"]:
        p = os.path.join(DATASET_ROOT, "Train", cls)
        sample_files = [f for f in os.listdir(p) if not f.startswith('.')][:5]
        print(f"\nInspecting 5 samples from {cls} folder:")
        for sf in sample_files:
            fp = os.path.join(p, sf)
            expected_num = class_to_idx[cls]
            expected_human = idx_to_class[expected_num]
            
            # Verify folder name consistency
            assigned_num = 0 if cls.lower() == "real" else 1
            assigned_human = idx_to_class[assigned_num]
            
            if assigned_num != expected_num or assigned_human != expected_human:
                print(f"  FAIL: {fp} -> assigned {assigned_human} != expected {expected_human}")
                passes = False
            else:
                print(f"  PASS: {sf} | Folder: {cls} | Numeric: {assigned_num} | Label: {assigned_human}")
                
    if passes:
        print("\nLABEL SANITY TEST: PASSED")
    else:
        print("\nLABEL SANITY TEST: FAILED")
    return passes

def audit_leakage(sample_size=1000):
    print("\n" + "=" * 60)
    print("4. CROSS-SPLIT DUPLICATE & LEAKAGE AUDIT")
    print("=" * 60)
    print(f"Computing SHA-256 on {sample_size} samples per split/class...")
    
    splits = ["Train", "Validation", "Test"]
    hashes = {}
    
    for split in splits:
        hashes[split] = {}
        for cls in ["Real", "Fake"]:
            p = os.path.join(DATASET_ROOT, split, cls)
            files = [f for f in os.listdir(p) if not f.startswith('.')][:sample_size]
            for f in files:
                fp = os.path.join(p, f)
                h = compute_sha256(fp)
                hashes[split][h] = (f, cls)
                
    # Check overlaps
    train_h = set(hashes["Train"].keys())
    val_h = set(hashes["Validation"].keys())
    test_h = set(hashes["Test"].keys())
    
    train_val_overlap = train_h.intersection(val_h)
    train_test_overlap = train_h.intersection(test_h)
    val_test_overlap = val_h.intersection(test_h)
    
    print(f"  Train <-> Validation exact duplicates in sample: {len(train_val_overlap)}")
    print(f"  Train <-> Test exact duplicates in sample:       {len(train_test_overlap)}")
    print(f"  Validation <-> Test exact duplicates in sample: {len(val_test_overlap)}")
    
    if len(train_test_overlap) > 0:
        print(f"  WARNING: Detected exact image duplicates between Train and Test in the raw dataset!")
        sample_dup = list(train_test_overlap)[0]
        print(f"    Example duplicate: Train {hashes['Train'][sample_dup]} vs Test {hashes['Test'][sample_dup]}")
        print("    -> Cross-split deduplication is mandatory before training!")

if __name__ == "__main__":
    audit_dataset()
    audit_class_mapping()
    audit_label_sanity()
    audit_leakage()
