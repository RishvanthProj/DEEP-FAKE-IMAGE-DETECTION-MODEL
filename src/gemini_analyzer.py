import os
import json
import logging
from PIL import Image
from dotenv import load_dotenv

# Load local environment secrets
load_dotenv()

logger = logging.getLogger("DeepfakeDetector")

GEMINI_ANALYSIS_PROMPT = """
You are an image-forensics assistant working alongside a trained deepfake classification model.

Analyze the uploaded image conservatively.

Your task is NOT to decide that an image is fake merely because it has been resized, compressed, sharpened, denoised, beautified, HDR-enhanced, color-corrected or processed by a smartphone.

First determine whether a valid human face exists.

Then inspect whether the FACE itself appears synthetically generated, face-swapped, reconstructed or substantially manipulated.

Distinguish:
1. authentic photograph
2. authentic photograph with benign computational enhancement (HDR, portrait mode, skin smoothing, noise reduction)
3. image with possible non-face AI editing (background removal, generative fill outside the face, sky replacement)
4. possible synthetic/manipulated face (AI-generated face, deepfake swap, diffusion face)
5. uncertain

Pay particular attention to:
- facial geometry
- skin microtexture
- eyes, iris, pupils, reflections
- eyelids, eyebrows
- nose, mouth, teeth
- ears and hair boundaries
- facial boundary and blending transitions
- lighting consistency and natural shadows
- local texture continuity
- compression consistency
- high-frequency texture
- unnatural repeated patterns
- face-background integration

Do not treat a single artifact as decisive.
Do not assume AI enhancement means deepfake.
Do not identify an image as deepfake only because it appears unusually sharp, smooth, noisy, compressed or enhanced.
A genuine smartphone photograph may contain significant computational photography artifacts while remaining a completely authentic capture.

Report uncertainty where evidence is insufficient.
If the person's FACE appears authentic but the image has background editing, object removal, color grading or enhancement, classify it as an authentic face with editing rather than a face deepfake.

You MUST respond ONLY with a valid, parseable JSON object matching this exact schema:
{
  "face_detected": true,
  "face_count": 1,
  "face_authenticity": "authentic|synthetic|uncertain",
  "overall_editing_status": "none_detected|benign_processing_possible|non_face_edit_possible|face_manipulation_possible|uncertain",
  "deepfake_assessment": "real_face|deepfake_face|uncertain",
  "synthetic_face_score": 0.0,
  "benign_processing_score": 0.0,
  "non_face_edit_score": 0.0,
  "face_manipulation_score": 0.0,
  "quality_score": 0.8,
  "image_generation_signs": [],
  "face_swap_signs": [],
  "benign_enhancement_signs": [],
  "non_face_edit_signs": [],
  "supporting_regions": [],
  "contradictory_evidence": [],
  "reasoning_summary": "1-2 sentence forensic summary",
  "confidence": 0.85,
  "recommendation": "REAL|DEEPFAKE|UNCERTAIN|NO_FACE"
}
"""

class GeminiForensicAnalyzer:
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        self.is_configured = False
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.is_configured = True
            except Exception as e:
                logger.warning(f"Failed to initialize Google GenAI client: {e}")
                self.is_configured = False

    def analyze_image(self, image: Image.Image) -> dict:
        """
        Sends image to Gemini for conservative visual forensics second-opinion.
        Returns a structured dictionary matching the schema.
        Never crashes; returns fallback if unavailable.
        """
        if not self.is_configured or self.client is None:
            return {
                "available": False,
                "error": "Gemini API key not configured or SDK initialization failed",
                "recommendation": "UNAVAILABLE"
            }

        try:
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Limit resolution for API speed/efficiency if very large
            img_to_send = image
            if max(image.size) > 1024:
                img_to_send = image.copy()
                img_to_send.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img_to_send, GEMINI_ANALYSIS_PROMPT]
            )

            text_resp = response.text.strip()
            # Clean possible markdown wrapping ```json ... ```
            if text_resp.startswith("```json"):
                text_resp = text_resp[7:]
            if text_resp.startswith("```"):
                text_resp = text_resp[3:]
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3]
            text_resp = text_resp.strip()

            parsed = json.loads(text_resp)
            parsed["available"] = True
            return parsed

        except Exception as e:
            logger.warning(f"Gemini analysis error: {e}")
            return {
                "available": False,
                "error": str(e),
                "recommendation": "UNAVAILABLE"
            }
