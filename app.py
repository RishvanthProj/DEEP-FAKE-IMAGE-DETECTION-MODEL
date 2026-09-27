import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import time

from src.predict import DeepfakePredictor
from src.dataset import get_dataset_stats
from src.image_forensics import generate_ela, generate_edge_map, generate_noise_residual

# 1. Page Config - No sidebar by default
st.set_page_config(
    page_title="Image Authenticity Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS to enforce the Forensic/Analytical look and remove Streamlit UI
st.markdown("""
<style>
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* Forensic Color Palette */
    .stApp {
        background-color: #0B0F14;
        color: #F3F4F6;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    
    /* Top Header */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #27303A;
        padding-bottom: 10px;
        margin-top: -40px;
        margin-bottom: 40px;
    }
    .top-header-left h1 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 1px;
        color: #F3F4F6;
    }
    .top-header-left p {
        margin: 0;
        font-size: 0.8rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .top-header-right {
        text-align: right;
    }
    .top-header-right h1 {
        margin: 0;
        font-size: 0.9rem;
        color: #9CA3AF;
        font-weight: normal;
    }
    .top-header-right p {
        margin: 0;
        font-size: 0.8rem;
        color: #22C55E;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F3F4F6;
        border-bottom: 1px solid #27303A;
        padding-bottom: 5px;
        margin-top: 40px;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Panels */
    .info-panel {
        background-color: #11161D;
        border: 1px solid #27303A;
        padding: 15px;
        border-radius: 4px;
    }
    
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        background-color: transparent !important;
    }
    .data-table tr, .data-table th, .data-table td {
        background-color: transparent !important;
    }
    .data-table td {
        padding: 10px 0 !important;
        border-bottom: 1px solid #27303A !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
    }
    .data-table td:first-child {
        color: #9CA3AF !important;
    }
    .data-table td:last-child {
        text-align: right !important;
        color: #F3F4F6 !important;
    }
    
    /* Results */
    .result-box-real {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid #22C55E;
        padding: 30px;
        text-align: center;
        border-radius: 4px;
    }
    .result-box-fake {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        padding: 30px;
        text-align: center;
        border-radius: 4px;
    }
    .result-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    .color-real { color: #22C55E; }
    .color-fake { color: #EF4444; }
    
    .confidence-text {
        font-size: 1.2rem;
        color: #F3F4F6;
        margin-bottom: 20px;
    }
    
    /* Footer */
    .footer {
        margin-top: 80px;
        padding-top: 20px;
        border-top: 1px solid #27303A;
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    return DeepfakePredictor()

def load_metrics():
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json", "r") as f:
            return json.load(f)
    return None

def main():
    predictor = get_predictor()
    model_status = "MODEL READY" if predictor.is_ready else "MODEL NOT TRAINED"
    status_color = "#22C55E" if predictor.is_ready else "#EF4444"

    # --- TOP HEADER ---
    st.markdown(f"""
    <div class="top-header">
        <div class="top-header-left">
            <h1>DEEPFAKE DETECTION</h1>
            <p>Digital Image Forensics</p>
        </div>
        <div class="top-header-right">
            <h1>ResNeXt50_32x4d</h1>
            <p style="color: {status_color};">{model_status}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- INITIAL LANDING SECTION ---
    col_intro, col_upload = st.columns([1, 1], gap="large")
    
    with col_intro:
        st.markdown("<h2 style='font-size: 1.8rem; margin-top:0;'>DEEPFAKE IMAGE DETECTION</h2>", unsafe_allow_html=True)
        st.markdown(
            "\"Machine-learning analysis of facial imagery for potential synthetic or manipulated content.\"",
            unsafe_allow_html=True
        )
        
        st.markdown("<div style='margin-top:40px;' class='section-title'>MODEL INFORMATION</div>", unsafe_allow_html=True)
        st.markdown("""
        <table class="data-table">
            <tr><td>Architecture</td><td>ResNeXt50_32x4d</td></tr>
            <tr><td>Task</td><td>Binary Classification</td></tr>
            <tr><td>Classes</td><td>REAL / DEEPFAKE</td></tr>
            <tr><td>Input</td><td>224 × 224 RGB</td></tr>
            <tr><td>Framework</td><td>PyTorch</td></tr>
        </table>
        """, unsafe_allow_html=True)
        
    with col_upload:
        # Save uploaded file temporarily for ELA analysis
        uploaded_file = st.file_uploader("DROP IMAGE HERE", type=["jpg", "jpeg", "png", "webp"], label_visibility="hidden")
        if uploaded_file is None:
            st.markdown("""
            <div style="text-align:center; padding:40px; border:1px dashed #27303A; color:#9CA3AF; margin-top:20px;">
                Upload JPG / JPEG / PNG / WEBP<br>Maximum supported file size: 200MB
            </div>
            """, unsafe_allow_html=True)

    # --- AFTER IMAGE UPLOAD ---
    if uploaded_file is not None:
        
        # Save temp file for ELA
        temp_img_path = "temp_upload.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        image = Image.open(uploaded_file)
        
        if not predictor.is_ready:
            st.error("ANALYSIS COULD NOT BE COMPLETED. Reason: Model unavailable.")
            return

        with st.spinner("ANALYZING IMAGE..."):
            res = predictor.predict(image)
            
        if "error" in res:
            st.error(f"ANALYSIS COULD NOT BE COMPLETED. Reason: {res['error']}")
            return

        st.markdown("<div class='section-title'>UPLOADED IMAGE</div>", unsafe_allow_html=True)
        col_img, col_det = st.columns([7, 5], gap="large")
        
        with col_img:
            st.image(image, use_container_width=True)
            
        with col_det:
            # Image Details
            img_np = np.array(image.convert('L'))
            blur_score = cv2.Laplacian(img_np, cv2.CV_64F).var()
            f_info = res.get("face_info", {"face_detected": False, "faces_count": 0, "confidence": 0.0})
            
            # 16. LOGGING
            import logging
            logging.info(
                f"Prediction Log - "
                f"filename: {uploaded_file.name}, "
                f"face_detected: {f_info.get('face_detected', False)}, "
                f"number_of_faces: {f_info.get('faces_count', 0)}, "
                f"face_confidence: {f_info.get('confidence', 0.0):.4f}, "
                f"classifier_executed: {res.get('prediction') != 'NO_FACE'}, "
                f"prediction: {res.get('prediction', 'NONE')}, "
                f"confidence: {res.get('confidence', 0.0):.4f}"
            )
            
            st.markdown("""
            <table class="data-table">
                <tr><td>Filename</td><td>{}</td></tr>
                <tr><td>Format</td><td>{}</td></tr>
                <tr><td>Resolution</td><td>{} × {}</td></tr>
                <tr><td>File Size</td><td>{:.1f} KB</td></tr>
                <tr><td>Color Mode</td><td>{}</td></tr>
                <tr><td>Face Detected</td><td>{}</td></tr>
                <tr><td>Number of Faces</td><td>{}</td></tr>
            </table>
            """.format(
                uploaded_file.name,
                image.format if image.format else "Unknown",
                image.size[0], image.size[1],
                uploaded_file.size / 1024,
                image.mode,
                "YES" if f_info["face_detected"] else "NO",
                f_info["faces_count"]
            ), unsafe_allow_html=True)

        st.markdown("<div class='section-title'>ANALYSIS RESULT</div>", unsafe_allow_html=True)
        
        pred_class = res["prediction"]
        
        if pred_class == "NO_FACE":
            st.markdown("""
            <div class="result-box-fake" style="background-color: rgba(156, 163, 175, 0.1); border: 1px solid #9CA3AF;">
                <div class="result-title" style="color: #F3F4F6; font-size: 2rem;">DEEPFAKE FACE NOT DETECTED</div>
                <div style="font-size: 1.2rem; color: #F3F4F6; margin-top: 15px; margin-bottom: 20px;">Try uploading a clear human face image.</div>
                <div style="color: #9CA3AF; font-size: 0.9rem;">No deepfake prediction was performed because no suitable human face was detected.</div>
            </div>
            """, unsafe_allow_html=True)
            return
            
        conf = res["confidence"] * 100
        p_real = res["prob_real"] * 100
        p_fake = res["prob_fake"] * 100
        
        if pred_class == "DEEPFAKE":
            st.markdown(f"""
            <div class="result-box-fake">
                <div class="result-title color-fake">DEEPFAKE</div>
                <div class="confidence-text">{conf:.1f}%<br><span style="font-size:0.8rem; color:#9CA3AF;">MODEL CONFIDENCE</span></div>
                <div style="width: 50%; margin: 0 auto; text-align:left;">
                    <div style="display:flex; justify-content:space-between; color:#9CA3AF; font-size:0.9rem;">
                        <span>P(REAL) {p_real:.1f}%</span><span>P(FAKE) {p_fake:.1f}%</span>
                    </div>
                    <div style="width:100%; height:8px; background-color:#27303A; margin-top:5px; border-radius:4px; overflow:hidden; display:flex;">
                        <div style="width:{p_real}%; height:100%; background-color:#22C55E;"></div>
                        <div style="width:{p_fake}%; height:100%; background-color:#EF4444;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div style='margin-top:15px; color:#9CA3AF; font-size:0.9rem;'><strong>MODEL INTERPRETATION:</strong> The model assigns a {p_fake:.1f}% probability to the DEEPFAKE class. The confidence reflects the classifier output for this image and does not constitute absolute proof of manipulation.</div>", unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class="result-box-real">
                <div class="result-title color-real">REAL</div>
                <div class="confidence-text">{conf:.1f}%<br><span style="font-size:0.8rem; color:#9CA3AF;">MODEL CONFIDENCE</span></div>
                <div style="width: 50%; margin: 0 auto; text-align:left;">
                    <div style="display:flex; justify-content:space-between; color:#9CA3AF; font-size:0.9rem;">
                        <span>P(REAL) {p_real:.1f}%</span><span>P(FAKE) {p_fake:.1f}%</span>
                    </div>
                    <div style="width:100%; height:8px; background-color:#27303A; margin-top:5px; border-radius:4px; overflow:hidden; display:flex;">
                        <div style="width:{p_real}%; height:100%; background-color:#22C55E;"></div>
                        <div style="width:{p_fake}%; height:100%; background-color:#EF4444;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div style='margin-top:15px; color:#9CA3AF; font-size:0.9rem;'><strong>MODEL INTERPRETATION:</strong> The model assigns a {p_real:.1f}% probability to the REAL class. This is the model's classification confidence and should not be interpreted as definitive proof of authenticity.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>FACE ANALYSIS</div>", unsafe_allow_html=True)
        if f_info["face_detected"]:
            b = [int(x) for x in f_info["bbox"]]
            st.markdown(f"Detected region: **x:** {b[0]} | **y:** {b[1]} | **width:** {b[2]-b[0]} | **height:** {b[3]-b[1]}")
        else:
            st.markdown("No distinct face region identified.")

        st.markdown("<div class='section-title'>MODEL EVIDENCE</div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.image(res["analysis_image"], caption="ORIGINAL (INPUT)", use_container_width=True)
        with col_c2:
            if res["gradcam_image"]:
                st.image(res["gradcam_image"], caption="GRAD-CAM HEATMAP", use_container_width=True)
            else:
                st.markdown("Not generated")
        with col_c3:
            if res["gradcam_image"]:
                st.image(res["gradcam_image"], caption="OVERLAY", use_container_width=True)
            else:
                st.markdown("Not generated")
                
        st.markdown("<div style='margin-top:15px; color:#9CA3AF; font-size:0.9rem;'>Highlighted regions indicate areas that contributed most strongly to the model's prediction.</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top:20px; font-weight:bold; color:#F3F4F6;'>WHY THE MODEL LEANED THIS WAY</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <table class="data-table">
            <tr><td style="padding-right:20px;">Prediction:</td><td style="color:#F3F4F6;">{pred_class}</td></tr>
            <tr><td style="padding-right:20px;">Probability:</td><td style="color:#F3F4F6;">{conf:.1f}%</td></tr>
            <tr><td style="padding-right:20px;">Strongest activation regions:</td><td style="color:#F3F4F6;">Face region / detected facial areas</td></tr>
            <tr><td style="padding-right:20px;">Evidence:</td><td style="color:#F3F4F6;">Grad-CAM visualization</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>SUPPLEMENTARY IMAGE FORENSICS</div>", unsafe_allow_html=True)
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            try:
                ela_img = generate_ela(temp_img_path)
                st.image(ela_img, caption="ERROR LEVEL ANALYSIS", use_container_width=True)
                st.markdown("<p style='font-size:0.8rem; color:#9CA3AF;'>Visualization of local compression differences. This is supporting evidence only and is not a definitive deepfake detector.</p>", unsafe_allow_html=True)
            except:
                st.markdown("ELA: Not available for this image format.")
                
            try:
                img_rgb = np.array(image.convert('RGB'))
                edge_img = generate_edge_map(img_rgb)
                st.image(edge_img, caption="EDGE MAP", use_container_width=True)
                st.markdown("<p style='font-size:0.8rem; color:#9CA3AF;'>Highlights abrupt structural transitions and splices.</p>", unsafe_allow_html=True)
            except:
                pass
                
        with col_f2:
            try:
                img_rgb = np.array(image.convert('RGB'))
                noise_img = generate_noise_residual(img_rgb)
                st.image(noise_img, caption="HIGH-FREQUENCY RESIDUAL", use_container_width=True)
                st.markdown("<p style='font-size:0.8rem; color:#9CA3AF;'>Highlights inconsistencies in sensor noise or GAN artifact patterns.</p>", unsafe_allow_html=True)
            except:
                st.markdown("Noise Residual: Not available.")

        st.markdown("<div class='section-title'>IMAGE QUALITY</div>", unsafe_allow_html=True)
        st.markdown("""
        <table class="data-table">
            <tr><td>Resolution</td><td>{} × {}</td></tr>
            <tr><td>Sharpness</td><td>{:.1f}</td></tr>
            <tr><td>Faces</td><td>{}</td></tr>
            <tr><td>Image Format</td><td>{}</td></tr>
        </table>
        """.format(
            image.size[0], image.size[1],
            blur_score,
            f_info["faces_count"],
            image.format
        ), unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#9CA3AF; margin-top:5px;'>Image quality can affect model reliability.</p>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>MODEL ANALYSIS</div>", unsafe_allow_html=True)
        st.markdown("""
        <table class="data-table">
            <tr><td>Architecture</td><td>ResNeXt50_32x4d</td></tr>
            <tr><td>Task</td><td>Binary classification</td></tr>
            <tr><td>Input</td><td>224 × 224 RGB</td></tr>
            <tr><td>Classes</td><td>REAL / DEEPFAKE</td></tr>
            <tr><td>Device</td><td>{}</td></tr>
            <tr><td>Inference Time</td><td>{:.1f} ms</td></tr>
        </table>
        """.format(str(predictor.device).upper(), res["inference_time_ms"]), unsafe_allow_html=True)

        st.markdown("<div class='section-title'>MODEL PERFORMANCE</div>", unsafe_allow_html=True)
        metrics = load_metrics()
        if metrics:
            st.markdown("""
            <table class="data-table">
                <tr><td>Accuracy</td><td>{:.4f}</td></tr>
                <tr><td>Precision</td><td>{:.4f}</td></tr>
                <tr><td>Recall</td><td>{:.4f}</td></tr>
                <tr><td>F1 Score</td><td>{:.4f}</td></tr>
                <tr><td>Specificity</td><td>{:.4f}</td></tr>
                <tr><td>ROC-AUC</td><td>{:.4f}</td></tr>
                <tr><td>PR-AUC</td><td>{:.4f}</td></tr>
            </table>
            """.format(
                metrics.get("accuracy", 0), metrics.get("precision", 0), 
                metrics.get("recall", 0), metrics.get("f1_score", 0),
                metrics.get("specificity", 0), metrics.get("roc_auc", 0),
                metrics.get("pr_auc", 0)
            ), unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.8rem; color:#9CA3AF; margin-top:5px;'>Test accuracy describes performance across the held-out evaluation dataset. The confidence shown above describes only the current uploaded image.</p>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-title'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
            col_cm1, col_cm2 = st.columns([1, 1])
            with col_cm1:
                if os.path.exists("results/confusion_matrix.png"):
                    st.image("results/confusion_matrix.png", use_container_width=True)
            with col_cm2:
                cm = metrics.get("confusion_matrix", {})
                st.markdown("""
                <table class="data-table" style="margin-top:40px;">
                    <tr><td>True Negatives (Actual REAL, Predicted REAL)</td><td>{}</td></tr>
                    <tr><td>False Positives (Actual REAL, Predicted FAKE)</td><td>{}</td></tr>
                    <tr><td>False Negatives (Actual FAKE, Predicted REAL)</td><td>{}</td></tr>
                    <tr><td>True Positives (Actual FAKE, Predicted FAKE)</td><td>{}</td></tr>
                </table>
                """.format(cm.get("TN",0), cm.get("FP",0), cm.get("FN",0), cm.get("TP",0)), unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem; color:#9CA3AF;'><strong>False Negative:</strong> A fake image incorrectly classified as real.<br><strong>False Positive:</strong> A real image incorrectly classified as fake.</p>", unsafe_allow_html=True)
                
            st.markdown("<div class='section-title'>ROC CURVE | PRECISION-RECALL</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.8rem; color:#9CA3AF;'>Charts will be automatically populated here upon full evaluation execution.</p>", unsafe_allow_html=True)
            
        else:
            st.markdown("<p style='color:#9CA3AF;'>No evaluation metrics found. Please run test evaluation.</p>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>TRAINING PERFORMANCE</div>", unsafe_allow_html=True)
        if os.path.exists("results/training_history.json"):
            with open("results/training_history.json", "r") as f:
                history = json.load(f)
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.write("Loss Curve")
                st.line_chart(pd.DataFrame({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}))
            with col_t2:
                st.write("Accuracy Curve")
                st.line_chart(pd.DataFrame({"Train Acc": history["train_acc"], "Val Acc": history["val_acc"]}))
        else:
            st.markdown("<p style='color:#9CA3AF;'>No training history found.</p>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>MODEL ERROR ANALYSIS</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#9CA3AF;'>Analysis dashboard populates during large-scale testing containing exact instances of model misclassification (False Positives and False Negatives).</p>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>TRAINING DATASET</div>", unsafe_allow_html=True)
        stats = get_dataset_stats()
        if stats:
            st.markdown("""
            <table class="data-table">
                <tr><td>Total Images</td><td>{}</td></tr>
                <tr><td>Real Images</td><td>{}</td></tr>
                <tr><td>Fake Images</td><td>{}</td></tr>
                <tr><td>Training Split</td><td>{}</td></tr>
                <tr><td>Validation Split</td><td>{}</td></tr>
                <tr><td>Test Split</td><td>{}</td></tr>
            </table>
            """.format(
                stats.get('total', 0),
                stats.get('real', 0),
                stats.get('fake', 0),
                stats.get('splits', {}).get('Train', 0),
                stats.get('splits', {}).get('Validation', 0),
                stats.get('splits', {}).get('Test', 0)
            ), unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("""
    <div class="footer">
        <p><strong>DEEPFAKE DETECTION</strong><br>Machine-learning based image authenticity analysis</p>
        <p>Model: ResNeXt50_32x4d</p>
        <p style="font-style:italic;">Predictions are model-based and should be interpreted as analytical evidence rather than absolute proof of authenticity.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
