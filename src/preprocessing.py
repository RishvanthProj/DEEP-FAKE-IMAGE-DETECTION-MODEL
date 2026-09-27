import torch
from torchvision import transforms
from PIL import Image

# ImageNet normalization standards
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

def get_train_transforms():
    """
    Transforms for training: resizing, mild augmentation, tensor conversion, normalization.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(10),
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
    Process a single PIL Image.
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

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_df, transform=None, use_face_crop=False, face_detector=None):
        self.df = manifest_df
        self.transform = transform
        self.use_face_crop = use_face_crop
        self.face_detector = face_detector
        
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
                # Attempt to get face crop
                face_crop, _ = self.face_detector.get_primary_face_crop(image)
                if face_crop is not None:
                    image = face_crop
                    
            if self.transform:
                image = self.transform(image)
                
            return image, label, img_path
            
        except Exception as e:
            # If an error occurs, return a blank tensor and the label
            # During training, we might need a better collate_fn to filter these out
            print(f"Error loading {img_path}: {e}")
            empty_tensor = torch.zeros((3, INPUT_SIZE, INPUT_SIZE))
            return empty_tensor, label, img_path
