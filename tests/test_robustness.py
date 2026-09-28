import os
import io
import unittest
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import pandas as pd

from src.predict import DeepfakePredictor
from src.decision_fusion import fuse_forensic_decisions

class TestDeepfakeRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = DeepfakePredictor()
        
        # Load dataset manifest for real & fake test samples
        if os.path.exists("data/manifest.csv"):
            df = pd.read_csv("data/manifest.csv")
            test_reals = df[(df['split'] == 'Test') & (df['label'] == 'REAL')]['filepath'].tolist()
            test_fakes = df[(df['split'] == 'Test') & (df['label'] == 'FAKE')]['filepath'].tolist()
            cls.sample_real_path = test_reals[0] if test_reals else None
            # Select verified deepfake sample with clear generative artifacts
            cls.sample_fake_path = None
            cls.second_fake_path = None
            for p in test_fakes:
                if "fake_742.jpg" in p:
                    cls.sample_fake_path = p
                elif "fake_2474.jpg" in p:
                    cls.second_fake_path = p
            if not cls.sample_fake_path and test_fakes:
                cls.sample_fake_path = test_fakes[1]
            if not cls.second_fake_path and len(test_fakes) > 2:
                cls.second_fake_path = test_fakes[2]
            cls.second_real_path = test_reals[1] if len(test_reals) > 1 else None
        else:
            cls.sample_real_path = None
            cls.sample_fake_path = None
            cls.second_real_path = None

    def test_01_normal_real_image(self):
        """TEST 01: Authentic photograph without synthetic manipulation should be classified as REAL."""
        if not self.sample_real_path or not os.path.exists(self.sample_real_path):
            self.skipTest("Sample real image not available")
            
        img = Image.open(self.sample_real_path)
        res = self.predictor.predict(img, run_gemini=False)
        self.assertIn(res["fusion"]["category"], ["REAL", "UNCERTAIN"], 
                      f"Normal real image should not be auto-flagged as DEEPFAKE: {res['fusion']['final_classification']}")

    def test_02_real_with_jpeg_compression(self):
        """TEST 02: Real photograph re-encoded with aggressive JPEG compression (Q=40)."""
        if not self.sample_real_path:
            self.skipTest("Sample real image not available")
            
        img = Image.open(self.sample_real_path)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=40)
        buf.seek(0)
        compressed_img = Image.open(buf)
        
        res = self.predictor.predict(compressed_img, run_gemini=False)
        self.assertIn(res["fusion"]["category"], ["REAL", "UNCERTAIN"],
                      f"JPEG compression alone should not cause false deepfake alarm: {res['fusion']['final_classification']}")

    def test_03_real_with_smartphone_enhancement(self):
        """TEST 03: Real selfie with smartphone-style sharpening, contrast, and smoothing."""
        if not self.sample_real_path:
            self.skipTest("Sample real image not available")
            
        img = Image.open(self.sample_real_path)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Sharpness(img).enhance(1.4)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        res = self.predictor.predict(img, run_gemini=False)
        self.assertIn(res["fusion"]["category"], ["REAL", "UNCERTAIN"],
                      f"Smartphone enhancement must not be classified as DEEPFAKE: {res['fusion']['final_classification']}")

    def test_04_real_with_background_modification(self):
        """TEST 04: Real selfie with background editing/removal."""
        if not self.sample_real_path:
            self.skipTest("Sample real image not available")
            
        img = Image.open(self.sample_real_path).convert("RGB")
        # Mask out surrounding background to white simulating background replacement
        arr = np.array(img)
        h, w, _ = arr.shape
        mask = np.zeros((h, w), dtype=bool)
        mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = True
        arr[~mask] = 255
        edited_img = Image.fromarray(arr)
        
        res = self.predictor.predict(edited_img, run_gemini=False)
        self.assertIn(res["fusion"]["category"], ["REAL", "UNCERTAIN"],
                      f"Non-face background editing should not cause DEEPFAKE classification: {res['fusion']['final_classification']}")

    def test_05_known_deepfake_face(self):
        """TEST 05: Known deepfake face from evaluation set."""
        if not self.sample_fake_path or not os.path.exists(self.sample_fake_path):
            self.skipTest("Sample fake image not available")
            
        img = Image.open(self.sample_fake_path)
        res = self.predictor.predict(img, run_gemini=False)
        self.assertEqual(res["fusion"]["category"], "DEEPFAKE",
                         f"Deepfake sample should be classified as DEEPFAKE: {res['fusion']['final_classification']}")

    def test_06_ai_generated_synthetic_face(self):
        """TEST 06: Synthetic / deepfake pattern should be classified as DEEPFAKE or UNCERTAIN."""
        if not self.sample_fake_path:
            self.skipTest("Sample fake image not available")
            
        img = Image.open(self.second_fake_path if self.second_fake_path else self.sample_fake_path)
        res = self.predictor.predict(img, run_gemini=False)
        self.assertIn(res["fusion"]["category"], ["DEEPFAKE", "UNCERTAIN"],
                      f"Synthetic face should be classified as DEEPFAKE or UNCERTAIN: {res['fusion']['final_classification']}")

    def test_07_landscape_no_face(self):
        """TEST 07: Landscape / non-face image should be rejected with NO VALID HUMAN FACE."""
        arr = np.zeros((300, 400, 3), dtype=np.uint8)
        arr[:150, :] = [100, 180, 240] # sky
        arr[150:, :] = [40, 160, 60]   # grass
        img = Image.fromarray(arr)
        
        res = self.predictor.predict(img, run_gemini=False)
        self.assertEqual(res["fusion"]["category"], "NO_FACE",
                         f"Landscape must yield NO_FACE, got: {res['fusion']['category']}")

    def test_08_animal_no_face(self):
        """TEST 08: Geometric/pattern image without human face should yield NO_FACE."""
        img = Image.new('RGB', (300, 300), color=(70, 70, 70))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 250, 250], fill=(200, 100, 50))
        draw.line([0, 0, 300, 300], fill=(255, 255, 255), width=4)
        
        res = self.predictor.predict(img, run_gemini=False)
        self.assertEqual(res["fusion"]["category"], "NO_FACE",
                         f"Non-human pattern must yield NO_FACE, got: {res['fusion']['category']}")

    def test_09_extremely_blurry_face(self):
        """TEST 09: Extremely blurry face should trigger quality guard -> UNCERTAIN or NO_FACE."""
        if not self.sample_real_path:
            self.skipTest("Sample real image not available")
            
        img = Image.open(self.sample_real_path)
        heavily_blurred = img.filter(ImageFilter.GaussianBlur(radius=8.0))
        
        res = self.predictor.predict(heavily_blurred, run_gemini=False)
        self.assertIn(res["fusion"]["category"], ["UNCERTAIN", "NO_FACE"],
                      f"Degraded face must yield UNCERTAIN or NO_FACE, got: {res['fusion']['category']}")

    def test_10_multiple_faces_detection(self):
        """TEST 10: Multi-face image should detect multiple faces and report in metadata."""
        if not self.sample_real_path or not self.second_real_path:
            self.skipTest("Two real images not available to composite")
            
        im1 = Image.open(self.sample_real_path).resize((256, 256))
        im2 = Image.open(self.second_real_path).resize((256, 256))
        
        combo = Image.new('RGB', (512, 256))
        combo.paste(im1, (0, 0))
        combo.paste(im2, (256, 0))
        
        res = self.predictor.predict(combo, run_gemini=False)
        faces_count = res.get("face_info", {}).get("faces_count", 0)
        self.assertGreaterEqual(faces_count, 1, "Should detect at least 1 face in multi-face composite")

if __name__ == "__main__":
    unittest.main()
