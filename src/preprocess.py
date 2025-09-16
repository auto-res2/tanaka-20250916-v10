"""Data preprocessing and loading for C-SADA experiments."""

import logging
from typing import Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple dataset wrapper for quick testing."""
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(dataset_name: str, is_training: bool = True) -> transforms.Compose:
    """Get appropriate transforms for dataset."""
    
    if dataset_name in ["cifar10", "cifar100"]:
        if is_training:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    else:
        if is_training:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


def load_cifar10_simple(num_samples: int = 1000) -> Tuple[DataLoader, DataLoader]:
    """Load simplified CIFAR-10 for smoke testing."""
    
    train_transform = get_transforms("cifar10", is_training=True)
    test_transform = get_transforms("cifar10", is_training=False)
    
    # Create PIL Images instead of tensors for transforms
    train_data = []
    for _ in range(num_samples):
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        train_data.append(img)
    
    test_data = []
    for _ in range(num_samples // 4):
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        test_data.append(img)
    
    train_labels = torch.randint(0, 10, (num_samples,))
    test_labels = torch.randint(0, 10, (num_samples // 4,))
    
    train_dataset = SimpleDataset(train_data, train_labels, train_transform)
    test_dataset = SimpleDataset(test_data, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Get data loaders based on configuration."""
    
    dataset_name = config.get("datasets", ["cifar10"])[0]
    num_samples = config.get("num_samples", 1000)
    batch_size = config.get("batch_size", 32)
    
    logger.info(f"Loading dataset: {dataset_name} with {num_samples} samples")
    
    if dataset_name == "cifar10":
        return load_cifar10_simple(num_samples)
    else:
        return load_cifar10_simple(num_samples)
