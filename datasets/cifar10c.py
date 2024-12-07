import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

template = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a corrupted photo of a {}.",
    "a distorted photo of a {}.",
    "a noisy photo of a {}.",
    "a low quality photo of a {}.",
    "a bad photo of a {}."
]

CIFAR10_CLASSNAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class CIFAR10C(Dataset):
    """CIFAR-10-C dataset for testing corruption robustness."""
    
    def __init__(self, root, corruption_type='gaussian_noise', severity=1, transform=None):
        """
        Args:
            root (str): Dataset root directory
            corruption_type (str): Type of corruption
            severity (int): Severity level (1-5)
            transform (callable, optional): Transform to apply to the image
        """
        self.transform = transform
        self.template = template
        self.classnames = CIFAR10_CLASSNAMES

        # Construct paths according to the dataset structure
        dataset_root = os.path.join(root, 'cifar10-c')
        severity_dir = os.path.join(dataset_root, 'corrupted', f'severity-{severity}')
        data_path = os.path.join(severity_dir, f'{corruption_type}.npy')
        label_path = os.path.join(dataset_root, 'origin', 'labels.npy')  # Labels are in origin directory

        print(f"Loading CIFAR-10-C data:")
        print(f"- Data file: {data_path}")
        print(f"- Label file: {label_path}")

        if not os.path.exists(data_path) or not os.path.exists(label_path):
            raise RuntimeError(
                f'Dataset files not found. Please check paths:\n'
                f'Data path: {data_path}\n'
                f'Label path: {label_path}'
            )

        # Load the data
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        print(f"Loaded {len(self.data)} images with shape {self.data.shape}")

        # Default transform if none provided - use CLIP's preprocessing
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a data point.
        
        Args:
            index (int): Index
        
        Returns:
            tuple: (image, label) where image is a tensor and label is the class label
        """
        img = self.data[index]
        label = self.labels[index]
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img)
        
        # Apply transforms to convert to tensor and normalize
        img = self.transform(img)
            
        return img, label 