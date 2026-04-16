"""
Data loading module.
Supports three experimental conditions:
  - full_data : full dataset, standard augmentation
  - low_data  : 10% random subset (simulates data-scarce regime)
  - noisy     : full dataset with additive Gaussian noise (noise_std)

Public API:
    set_seed(seed)
    get_transforms(img_size, condition, noise_std) -> transforms.Compose
    ImageFolderFlat(root, transform)              -> Dataset
    get_dataloader(data_dir, condition, batch_size, img_size,
                   seed, num_workers, noise_std)  -> DataLoader
"""

import os
import random
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Covers Python random, NumPy, PyTorch (CPU + CUDA), and enables
    deterministic cuDNN mode.

    Args:
        seed: integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class AddGaussianNoise:
    """Add zero-mean Gaussian noise to a tensor image in [-1, 1]."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, -1.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(img_size: int = 64,
                   condition: str = 'full_data',
                   noise_std: float = 0.1) -> transforms.Compose:
    """
    Build a torchvision transform pipeline for the given condition.

    All conditions share the same base pipeline:
        Resize -> CenterCrop -> ToTensor -> Normalize([-1, 1])

    The ``noisy`` condition additionally appends an AddGaussianNoise step.

    Args:
        img_size:   target spatial size (square)
        condition:  one of {'full_data', 'low_data', 'noisy'}
        noise_std:  std of Gaussian noise (only used when condition='noisy')

    Returns:
        transforms.Compose
    """
    base = [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),   # maps [0,1] -> [-1,1]
    ]

    if condition == 'noisy':
        base.append(AddGaussianNoise(std=noise_std))

    return transforms.Compose(base)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageFolderFlat(Dataset):
    """
    Flat image folder dataset.

    Loads all images directly from ``root`` (and optionally its
    subdirectories) without requiring class-named sub-folders.
    Compatible with torchvision's ImageFolder API for evaluation code.

    Supported extensions: .jpg  .jpeg  .png  (case-insensitive)

    Args:
        root:      path to the directory containing images
        transform: optional torchvision transform pipeline
    """

    EXTENSIONS = ('.jpg', '.jpeg', '.png')

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._collect_images(root)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in '{root}'. "
                f"Expected files with extensions {self.EXTENSIONS}."
            )

    @staticmethod
    def _collect_images(root: str):
        paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png',
                    '*.JPG', '*.JPEG', '*.PNG'):
            paths.extend(sorted(glob.glob(
                os.path.join(root, '**', ext), recursive=True
            )))
        # Deduplicate while preserving order (glob may return duplicates on
        # case-insensitive file systems)
        seen = set()
        unique = []
        for p in paths:
            key = os.path.normcase(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically.
    Must be a module-level function to be picklable on Windows."""
    # _WORKER_SEED is set by get_dataloader before creating the DataLoader
    base_seed = _WORKER_BASE_SEED
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


_WORKER_BASE_SEED: int = 42  # overwritten by get_dataloader at runtime


def get_dataloader(data_dir: str,
                   condition: str = 'full_data',
                   batch_size: int = 64,
                   img_size: int = 64,
                   seed: int = 42,
                   num_workers: int = 4,
                   noise_std: float = 0.1) -> DataLoader:
    """
    Build a DataLoader for the given experimental condition.

    Conditions
    ----------
    full_data : all available images, standard preprocessing
    low_data  : 10% random subset drawn with ``seed`` (reproducible)
    noisy     : all images + additive Gaussian noise (std=``noise_std``)

    Args:
        data_dir:    path to directory containing images
        condition:   one of {'full_data', 'low_data', 'noisy'}
        batch_size:  mini-batch size
        img_size:    spatial size (square) after resizing
        seed:        random seed for subset selection and worker init
        num_workers: DataLoader worker processes
        noise_std:   Gaussian noise std (only used for 'noisy' condition)

    Returns:
        torch.utils.data.DataLoader
    """
    if condition not in ('full_data', 'low_data', 'noisy'):
        raise ValueError(
            f"Unknown condition '{condition}'. "
            "Choose from 'full_data', 'low_data', 'noisy'."
        )

    transform = get_transforms(img_size=img_size,
                                condition=condition,
                                noise_std=noise_std)

    dataset = ImageFolderFlat(root=data_dir, transform=transform)

    if condition == 'low_data':
        # 10% random subset, reproducible via seed
        n_total = len(dataset)
        n_subset = max(1, int(n_total * 0.1))
        rng = np.random.RandomState(seed)
        indices = rng.choice(n_total, n_subset, replace=False).tolist()
        dataset = Subset(dataset, indices)
        print(f"[low_data] Using {n_subset}/{n_total} images "
              f"({100 * n_subset / n_total:.1f}%)")
    else:
        print(f"[{condition}] Using {len(dataset)} images")

    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = seed

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    return loader
