import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

class FaceDetector:
    def __init__(self, device='cpu'):
        # PyTorch MPS has an adaptive pool bug with MTCNN. Force CPU if MPS.
        if str(device) == 'mps':
            device = 'cpu'
        self.device = device
        self.mtcnn_all = MTCNN(keep_all=True, device=device)
        self.mtcnn_single = MTCNN(keep_all=False, device=device)
        
    def detect_faces(self, image: Image.Image):
        """
        Detect all faces in an image.
        Returns bounding boxes and probabilities.
        """
        # MTCNN expects RGB image
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        boxes, probs = self.mtcnn_all.detect(image)
        if boxes is None:
            return [], []
        return boxes.tolist(), probs.tolist()

    def get_primary_face_crop(self, image: Image.Image, margin=20):
        """
        Detects the primary face and returns a cropped PIL Image of the face.
        If no face is detected, returns None.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        boxes, probs = self.mtcnn_single.detect(image)
        if boxes is None or len(boxes) == 0:
            return None, None
            
        # Get the first (primary) face bounding box
        box = boxes[0]
        
        # Add margin
        width, height = image.size
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)
        
        face_crop = image.crop((x1, y1, x2, y2))
        return face_crop, [x1, y1, x2, y2]
