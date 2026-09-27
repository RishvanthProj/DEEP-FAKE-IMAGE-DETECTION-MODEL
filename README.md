# Deepfake Image Detection 🔍

A complete Machine Learning pipeline and Web Application for detecting Deepfake images, built with PyTorch and Streamlit.
This project implements the core feature-extraction concepts of ResNeXt with a binary classification head for static images, complete with face detection, cross-split duplicate removal, explainability (Grad-CAM++), and comprehensive metrics reporting.

## Features
- **Robust Preprocessing**: Uses MTCNN for reliable face detection and cropping.
- **Data Auditing**: Computes SHA-256 for all images and automatically drops cross-split duplicates.
- **Model Architecture**: Uses a ResNeXt50_32x4d backbone with a binary classification head.
- **Explainability**: Implements Grad-CAM++ to highlight regions the model focuses on.
- **Professional Dashboard**: Dark-themed forensic dashboard built with Streamlit for inference, diagnostics, and metrics.

## Project Structure
- `src/dataset.py`: Dataset discovery and split validation.
- `src/preprocessing.py`: Image transforms and PyTorch Dataset class.
- `src/face_detection.py`: MTCNN wrapper.
- `src/model.py`: ResNeXt50 architecture setup.
- `src/train.py`: Training pipeline (with frozen backbone head training + fine-tuning).
- `src/evaluate.py`: Metrics (Accuracy, F1, ROC-AUC) and Confusion Matrix.
- `src/predict.py`: Inference pipeline with uncertainty zones and Grad-CAM.
- `src/explainability.py`: Grad-CAM++ implementation.
- `app.py`: Streamlit Web Application.

## Getting Started

1. **Install requirements:**
```bash
python -m pip install -r requirements.txt
```

2. **Discover dataset and check for leakage (creates `data/manifest.csv`):**
```bash
PYTHONPATH=. python -m src.dataset
```

3. **Train the model:**
```bash
PYTHONPATH=. python src/train.py --batch-size 32 --epochs-head 5 --epochs-finetune 10 --use-face-crop
```
*(To run a quick smoke test instead of full training, add `--smoke-test`)*

4. **Launch the web application:**
```bash
PYTHONPATH=. streamlit run app.py
```
