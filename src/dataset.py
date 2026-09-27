import os
import glob
import hashlib
import pandas as pd
from PIL import Image
from src.utils import setup_logging

logger = setup_logging()

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def discover_dataset(train_dir, val_dir, test_dir, output_manifest="data/manifest.csv"):
    logger.info("Discovering dataset from explicit paths")
    
    records = []
    
    # Map possible class names to standard ones
    class_mapping = {
        "real": "REAL",
        "fake": "FAKE",
        "deepfake": "FAKE"
    }
    
    directories = {
        "Train": train_dir,
        "Validation": val_dir,
        "Test": test_dir
    }
    
    for split, data_dir in directories.items():
        if not os.path.exists(data_dir):
            logger.error(f"Directory {data_dir} does not exist!")
            continue
            
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    filepath = os.path.join(root, file)
                    
                    # Deduce label from folder name
                    folder_name = os.path.basename(root)
                    label = "Unknown"
                    if folder_name.lower() in class_mapping:
                        label = class_mapping[folder_name.lower()]
                    
                    if label == "Unknown":
                        continue
                    
                    try:
                        with Image.open(filepath) as img:
                            width, height = img.size
                            channels = len(img.getbands())
                    except Exception as e:
                        logger.warning(f"Could not read {filepath}: {e}")
                        continue
                        
                    file_size = os.path.getsize(filepath)
                    sha256 = compute_sha256(filepath)
                    
                    records.append({
                        "filepath": filepath,
                        "filename": file,
                        "label": label,
                        "split": split,
                        "width": width,
                        "height": height,
                        "channels": channels,
                        "file_size": file_size,
                        "sha256": sha256
                    })
                
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        logger.error("No valid dataset images found!")
        return None
        
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['sha256'], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {len(duplicates)} duplicate images based on SHA-256")
        
        # Cross split leakage
        dup_groups = duplicates.groupby('sha256')
        for sha, group in dup_groups:
            splits = group['split'].unique()
            if len(splits) > 1:
                logger.error(f"CROSS-SPLIT LEAKAGE DETECTED! SHA256: {sha} found in splits: {splits}")
                
        # Remove duplicates, keeping the first occurrence (preferably Train)
        # Custom sort to prefer keeping Train, then Validation, then Test
        split_order = {'Train': 0, 'Validation': 1, 'Test': 2}
        df['split_order'] = df['split'].map(split_order)
        df = df.sort_values('split_order').drop_duplicates(subset=['sha256'], keep='first')
        df = df.drop(columns=['split_order'])
        logger.info(f"Removed duplicates. Final dataset size: {len(df)}")
    
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    df.to_csv(output_manifest, index=False)
    logger.info(f"Dataset manifest saved to {output_manifest}")
    
    return df

def get_dataset_stats(manifest_path="data/manifest.csv"):
    if not os.path.exists(manifest_path):
        return None
    
    df = pd.read_csv(manifest_path)
    stats = {
        "total": len(df),
        "real": len(df[df['label'] == 'REAL']),
        "fake": len(df[df['label'] == 'FAKE']),
        "splits": df['split'].value_counts().to_dict(),
        "classes_by_split": df.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict('index')
    }
    return stats

if __name__ == "__main__":
    discover_dataset("data")
