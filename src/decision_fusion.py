import numpy as np

def fuse_forensic_decisions(
    ml_prediction: str,
    ml_prob_real: float,
    ml_prob_fake: float,
    ml_calibrated_prob_fake: float,
    tta_mean_fake: float,
    tta_std_fake: float,
    tta_stability: str,
    face_info: dict,
    gemini_result: dict = None,
    threshold_fake: float = 0.55,
    threshold_real: float = 0.45
) -> dict:
    """
    Conservative decision fusion layer combining ResNeXt-50 predictions,
    Test-Time Augmentation (TTA) stability metrics, face validation,
    and Gemini auxiliary multimodal second-opinion.
    
    Returns structured final decision with 4-class classification:
    - REAL / AUTHENTIC FACE
    - DEEPFAKE / SYNTHETIC FACE
    - UNCERTAIN
    - NO VALID HUMAN FACE
    """
    # 1. Gate: Face Detection Validation
    if not face_info.get("face_detected", False) or face_info.get("faces_count", 0) == 0:
        return {
            "final_classification": "NO VALID HUMAN FACE",
            "category": "NO_FACE",
            "face_status": "Not Detected",
            "edit_status": "none_detected",
            "final_confidence": 0.0,
            "explanation": "No valid human face was detected. Please upload a clear face image.",
            "ml_status": ml_prediction,
            "gemini_status": "N/A",
            "tta_stability": tta_stability,
            "is_certain": True
        }

    # 2. Gate: Face Quality Check
    face_confidence = face_info.get("confidence", 1.0)
    face_quality = face_info.get("face_quality", 1.0)
    if face_confidence < 0.60 or face_quality < 0.20:
        return {
            "final_classification": "UNCERTAIN",
            "category": "UNCERTAIN",
            "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
            "edit_status": "uncertain",
            "final_confidence": 0.50,
            "explanation": "The detected face quality is insufficient or heavily degraded. Analysis cannot reliably establish authenticity.",
            "ml_status": ml_prediction,
            "gemini_status": gemini_result.get("recommendation", "N/A") if gemini_result else "N/A",
            "tta_stability": tta_stability,
            "is_certain": False
        }

    # Use calibrated probability for decision
    prob_fake = ml_calibrated_prob_fake
    prob_real = 1.0 - prob_fake
    has_gemini = gemini_result is not None and gemini_result.get("available", False)

    # 3. Decision Logic with Gemini Auxiliary Analysis
    if has_gemini:
        gemini_rec = gemini_result.get("recommendation", "UNCERTAIN").upper()
        gemini_auth = gemini_result.get("face_authenticity", "uncertain").lower()
        gemini_edit = gemini_result.get("overall_editing_status", "none_detected")
        gemini_conf = gemini_result.get("confidence", 0.70)
        gemini_manip_score = gemini_result.get("face_manipulation_score", 0.0)
        gemini_benign_score = gemini_result.get("benign_processing_score", 0.0)

        # Case A: Strong Dual Consensus on DEEPFAKE
        if (prob_fake >= threshold_fake) and (gemini_auth == "synthetic" or gemini_manip_score > 0.60):
            final_conf = (prob_fake + gemini_conf) / 2.0
            return {
                "final_classification": "DEEPFAKE / SYNTHETIC FACE",
                "category": "DEEPFAKE",
                "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
                "edit_status": "probable_face_manipulation",
                "final_confidence": final_conf,
                "explanation": "Both the trained classifier and secondary visual analysis identified patterns associated with synthetic or manipulated facial content. The result is a model-based assessment, not absolute proof.",
                "ml_status": ml_prediction,
                "gemini_status": "Synthetic Face Evidence",
                "tta_stability": tta_stability,
                "is_certain": True
            }

        # Case B: Authentic Face with Benign Smartphone Processing or Non-Face Edit
        # (Resolves False Positives when smartphone HDR / beautification triggers ML)
        if gemini_auth == "authentic" and (gemini_benign_score > 0.40 or gemini_edit in ["benign_processing_possible", "non_face_edit_possible"]):
            if prob_fake < 0.75 or tta_stability == "LOW":
                final_conf = max(prob_real, gemini_conf)
                edit_label = "Benign Enhancement / Filter" if gemini_edit == "benign_processing_possible" else "Non-Face Edit"
                return {
                    "final_classification": "REAL / AUTHENTIC FACE",
                    "category": "REAL",
                    "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
                    "edit_status": gemini_edit,
                    "final_confidence": final_conf,
                    "explanation": f"Image-processing artifacts were observed consistent with normal smartphone computational photography, enhancement ({edit_label}), or compression. No sufficient evidence of synthetic facial manipulation was established.",
                    "ml_status": ml_prediction,
                    "gemini_status": f"Authentic Face ({edit_label})",
                    "tta_stability": tta_stability,
                    "is_certain": True
                }

        # Case C: Both Agree on REAL / AUTHENTIC
        if (prob_fake < threshold_fake) and (gemini_auth == "authentic"):
            final_conf = (prob_real + gemini_conf) / 2.0
            return {
                "final_classification": "REAL / AUTHENTIC FACE",
                "category": "REAL",
                "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
                "edit_status": gemini_edit,
                "final_confidence": final_conf,
                "explanation": "Your image contains a detected human face. The primary classifier did not find strong evidence associated with synthetic facial manipulation. The secondary visual analysis also found no sufficient evidence of a synthetic face. Minor image-processing artifacts may be consistent with normal smartphone photography.",
                "ml_status": ml_prediction,
                "gemini_status": "Authentic Face",
                "tta_stability": tta_stability,
                "is_certain": True
            }

        # Case D: Ambiguity or Direct Contradiction
        # If ML strongly says Deepfake (>0.80) but Gemini says Authentic without benign tags, or vice versa
        if tta_stability == "LOW" or (abs(prob_fake - 0.50) < 0.10):
            return {
                "final_classification": "UNCERTAIN",
                "category": "UNCERTAIN",
                "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
                "edit_status": "uncertain",
                "final_confidence": 0.50,
                "explanation": "The available evidence is contradictory or insufficient across multiple visual assessments. The system will not force a REAL or DEEPFAKE decision.",
                "ml_status": ml_prediction,
                "gemini_status": gemini_result.get("reasoning_summary", "Ambiguous evidence"),
                "tta_stability": tta_stability,
                "is_certain": False
            }

    # 4. Standalone ML Decision (when Gemini is unavailable)
    if tta_stability == "LOW":
        return {
            "final_classification": "UNCERTAIN",
            "category": "UNCERTAIN",
            "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
            "edit_status": "uncertain",
            "final_confidence": 0.50,
            "explanation": "Test-Time Augmentation revealed high prediction instability across mild image variations. The decision is withheld as UNCERTAIN.",
            "ml_status": ml_prediction,
            "gemini_status": "Secondary Analysis Unavailable",
            "tta_stability": tta_stability,
            "is_certain": False
        }

    if prob_fake >= threshold_fake:
        return {
            "final_classification": "DEEPFAKE / SYNTHETIC FACE",
            "category": "DEEPFAKE",
            "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
            "edit_status": "probable_face_manipulation",
            "final_confidence": prob_fake,
            "explanation": "The trained classifier identified patterns associated with synthetic or manipulated facial content. The result is a model-based assessment, not absolute proof.",
            "ml_status": "DEEPFAKE",
            "gemini_status": "Secondary Analysis Unavailable",
            "tta_stability": tta_stability,
            "is_certain": True
        }
    elif prob_fake <= threshold_real:
        return {
            "final_classification": "REAL / AUTHENTIC FACE",
            "category": "REAL",
            "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
            "edit_status": "none_detected",
            "final_confidence": prob_real,
            "explanation": "The primary classifier did not find strong evidence of synthetic facial manipulation. Minor artifacts may be consistent with normal smartphone camera processing.",
            "ml_status": "REAL",
            "gemini_status": "Secondary Analysis Unavailable",
            "tta_stability": tta_stability,
            "is_certain": True
        }
    else:
        return {
            "final_classification": "UNCERTAIN",
            "category": "UNCERTAIN",
            "face_status": f"Detected ({face_info.get('faces_count', 1)} face(s))",
            "edit_status": "uncertain",
            "final_confidence": 0.50,
            "explanation": "Model confidence is within the boundary of uncertainty. Further forensic evidence is required.",
            "ml_status": ml_prediction,
            "gemini_status": "Secondary Analysis Unavailable",
            "tta_stability": tta_stability,
            "is_certain": False
        }
