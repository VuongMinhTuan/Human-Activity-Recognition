from torchvision import transforms as T
from typing import Tuple



# Default transformation
def DEFAULT(img_size: Tuple = (224, 224) | list | int):
    return T.Compose([
        T.ToTensor(),
        T.Resize(img_size),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


# Convert pytorch tensor to PIL image and apply set of transformations
def ToPIL(img_size: Tuple = (224, 224) | list | int):
    return T.RandomOrder(T.Compose([
        T.ToPILImage(),
        T.Resize(img_size),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]))


# Customize and apply set of transformations
def Customize(img_size: Tuple = (224, 224) | list | int):
    return T.Compose([
        T.RandomResizedCrop(img_size, antialias=True),
        T.RandomHorizontalFlip(p=0.69),
        T.RandomRotation(96),
        T.GaussianBlur(9.6),
        T.RandomPerspective(0.69, p=0.69),
        T.RandomAutocontrast(p=0.69),
        T.RandomAdjustSharpness(9.6, p=0.69),
        T.RandomAffine(degrees=69, translate=(0.6, 0.9)),
        T.RandomEqualize(p=0.69),
        T.ColorJitter(brightness=0.69, contrast=0.69, saturation=0.69, hue=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])