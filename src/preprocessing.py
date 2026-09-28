import io
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

# ImageNet normalization standards
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

class SmartphoneAugmentation:
    """
    Controlled hard-negative augmentation representing genuine smartphone photography:
    - JPEG recompression (Q=45-92)
    - Mild skin smoothing / bilateral blur (radius 0.4-1.2)
    - Camera sharpening / edge enhancement (1.2x - 1.8x)
    - HDR-like exposure and contrast shifts (0.9x - 1.25x)
    - Mild sensor noise
    """
    def __init__(self, p=0.6):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        
        img = img.copy()
        aug_type = random.choice(['jpeg', 'smooth', 'sharpen', 'hdr', 'noise', 'combo'])
        
        if aug_type in ['jpeg', 'combo']:
            quality = random.randint(45, 90)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            
        if aug_type in ['smooth', 'combo']:
            radius = random.uniform(0.4, 1.1)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
        if aug_type in ['sharpen', 'combo']:
            factor = random.uniform(1.2, 1.7)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(factor)
            
        if aug_type in ['hdr', 'combo']:
            c_factor = random.uniform(0.9, 1.2)
            b_factor = random.uniform(0.9, 1.15)
            img = ImageEnhance.Contrast(img).enhance(c_factor)
            img = ImageEnhance.Brightness(img).enhance(b_factor)
            
        if aug_type == 'noise':
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(2, 7), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            
        return img

def get_train_transforms():
    """
    Standard geometric & photometric transforms for training.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.82, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def get_eval_transforms():
    """
    Transforms for validation and testing: resizing, center crop, tensor conversion, normalization.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def get_predict_transforms():
    """
    Transforms for single image prediction (same as eval).
    """
    return get_eval_transforms()

def process_image(image: Image.Image, transform_type='predict'):
    """
    Process a single PIL Image into a model input tensor.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    if transform_type == 'train':
        transform = get_train_transforms()
    elif transform_type == 'eval':
        transform = get_eval_transforms()
    else:
        transform = get_predict_transforms()
        
    tensor = transform(image)
    return tensor.unsqueeze(0) # Add batch dimension

def get_tta_batch(image: Image.Image, device='cpu'):
    """
    Generates a batch of mild Test-Time Augmentation (TTA) tensors:
    1. Original (Resize 256, CenterCrop 224)
    2. Horizontal Flip
    3. Mild Brightness Boost (+5%)
    4. Mild Contrast Boost (+5%)
    5. Slight Scale/Crop (Resize 248, CenterCrop 224)
    Returns: torch.Tensor of shape (5, 3, 224, 224) on specified device.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    base_eval = get_eval_transforms()
    
    # 1. Base
    t1 = base_eval(image)
    
    # 2. Horizontal Flip
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    t2 = base_eval(flipped)
    
    # 3. Mild Brightness (+5%)
    brighter = ImageEnhance.Brightness(image).enhance(1.05)
    t3 = base_eval(brighter)
    
    # 4. Mild Contrast (+5%)
    contrasted = ImageEnhance.Contrast(image).enhance(1.05)
    t4 = base_eval(contrasted)
    
    # 5. Slight Scale Variation (248x248 resized, center crop 224)
    t5_transform = transforms.Compose([
        transforms.Resize((248, 248)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    t5 = t5_transform(image)
    
    batch = torch.stack([t1, t2, t3, t4, t5]).to(device)
    return batch

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_df, transform=None, use_face_crop=False, face_detector=None, robust_augment_real=False):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.use_face_crop = use_face_crop
        self.face_detector = face_detector
        self.robust_augment_real = robust_augment_real
        self.smartphone_aug = SmartphoneAugmentation(p=0.6) if robust_augment_real else None
        
        # Label encoding: REAL = 0, FAKE = 1
        self.class_to_idx = {"REAL": 0, "FAKE": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label_str = row['label']
        label = self.class_to_idx[label_str]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.use_face_crop and self.face_detector is not None:
                face_crop, _ = self.face_detector.get_primary_face_crop(image)
                if face_crop is not None:
                    image = face_crop
            
            # Apply smartphone hard-negative augmentation specifically to REAL images during training
            if self.robust_augment_real and label == 0 and self.smartphone_aug is not None:
                image = self.smartphone_aug(image)
                    
            if self.transform:
                image = self.transform(image)
                
            return image, label, img_path
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            empty_tensor = torch.zeros((3, INPUT_SIZE, INPUT_SIZE))
            return empty_tensor, label, img_path
