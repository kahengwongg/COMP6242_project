"""
Data loading module.
Supports three experimental conditions: full_data, low_data, noisy
"""

import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np


class ImageFolderFlat(Dataset):
    """Generic image dataset that recursively finds all images in a directory."""
    
    def __init__(self, root_dir, transform=None, add_noise=False, noise_std=0.1):
        """
        Args:
            root_dir: dataset root directory
            transform: image transforms
            add_noise: whether to add noise
            noise_std: noise standard deviation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(
                self._find_files(root_dir, ext)
            )
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {root_dir}")
        
        print(f"Found {len(self.image_files)} images")
    
    def _find_files(self, root_dir, pattern):
        """Recursively find files"""
        import glob
        return sorted(glob.glob(os.path.join(root_dir, '**', pattern), recursive=True))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Add noise (only during training)
        if self.add_noise:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            # Ensure values remain in [-1, 1] range
            image = torch.clamp(image, -1, 1)
        
        return image


def get_transforms(img_size=64, augment=False):
    """
    Get image transforms.
    
    Args:
        img_size: target image size
        augment: whether to use data augmentation (random horizontal flip)
    
    Returns:
        torchvision.transforms.Compose
    """
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ]
    
    if augment:
        # Insert after Resize, before ToTensor
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
    
    return transforms.Compose(transform_list)


def _worker_init_fn(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.dataset  # unused; we use the base seed below
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def get_dataloader(
    data_dir,
    condition='full_data',
    batch_size=64,
    img_size=64,
    seed=42,
    num_workers=4,
    noise_std=0.1
):
    """
    Get data loader.
    
    Args:
        data_dir: dataset directory
        condition: experimental condition ('full_data', 'low_data', 'noisy')
        batch_size: batch size
        img_size: image size
        seed: random seed
        num_workers: number of data loading workers
        noise_std: noise standard deviation (only effective for noisy condition)
    
    Returns:
        DataLoader
    """
    # Determine whether to use data augmentation
    # Only full_data condition uses data augmentation
    use_augment = (condition == 'full_data')
    
    # Determine whether to add noise
    add_noise = (condition == 'noisy')
    
    # Create transforms
    transform = get_transforms(img_size, augment=use_augment)
    
    # Create dataset
    dataset = ImageFolderFlat(
        root_dir=data_dir,
        transform=transform,
        add_noise=add_noise,
        noise_std=noise_std
    )
    
    # Process dataset based on condition
    if condition == 'low_data':
        # Use fixed seed to sample 10% of data
        # Ensures all models use the same data subset under low_data condition
        total_size = len(dataset)
        subset_size = total_size // 10
        
        # Use fixed seed to generate indices
        rng = np.random.RandomState(seed)
        indices = rng.choice(total_size, subset_size, replace=False)
        indices = indices.tolist()
        
        dataset = Subset(dataset, indices)
        print(f"Low data mode: using {len(indices)} images (10%)")
    
    elif condition == 'noisy':
        # No data augmentation under noisy condition
        print(f"Noisy mode: noise std = {noise_std}")
    
    else:  # full_data
        print(f"Full data mode: using all {len(dataset)} images")
    
    # Create a seeded generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    
    return dataloader


def set_seed(seed):
    """Set global random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == '__main__':
    # Test data loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/anime_faces',
                        help='Dataset directory')
    parser.add_argument('--condition', type=str, default='full_data',
                        choices=['full_data', 'low_data', 'noisy'],
                        help='Experimental condition')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        condition=args.condition,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    print(f"\nData loader created successfully!")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test loading one batch
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")