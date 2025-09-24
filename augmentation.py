import torchvision.transforms as transforms
import torch

def get_train_transforms(input_size, random_crop_scale=(0.8, 1.0),
                         color_jitter_params=(0.2, 0.2, 0.2, 0.1),
                         affine_degrees=15, affine_translate=(0.1, 0.1),
                         affine_scale=(0.9, 1.1), affine_shear=10):
    """
    Build training augmentations.
    
    Args:
        input_size (tuple): target image size (H, W)
        random_crop_scale (tuple): min/max scale for RandomResizedCrop
        color_jitter_params (tuple): (brightness, contrast, saturation, hue)
        affine_degrees (float): max rotation
        affine_translate (tuple): translation fraction (h, w)
        affine_scale (tuple): min/max scaling
        affine_shear (float or tuple): shear angle
    
    Returns:
        torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=random_crop_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(*color_jitter_params),
        transforms.RandomAffine(degrees=affine_degrees,
                                translate=affine_translate,
                                scale=affine_scale,
                                shear=affine_shear),
        transforms.ToTensor(),
        normalize
    ])


def get_val_transforms(input_size):
    """Validation/test transforms (no augmentation)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize
    ])
