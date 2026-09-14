import torchvision.transforms as T
import torch

def get_train_transforms(image_size: int = 224):
    """
    Standard training transforms with data augmentation.
    Includes spatial and color perturbations to help generalization.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size: int = 224):
    """
    Standard validation transforms (deterministic).
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_inference_transforms(image_size: int = 224):
    """
    Inference transforms, similar to validation but may include TTA steps if configured later.
    """
    return get_val_transforms(image_size)
