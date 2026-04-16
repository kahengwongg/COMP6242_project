"""
Data download script using kagglehub.

Downloads datasets from Kaggle and places them in the expected directories.

Usage:
    python -m utils.download_data                  # Anime Face Dataset (default)
    python -m utils.download_data --dataset celeba # CelebA Dataset
"""

import argparse
import os
import shutil
import glob


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASETS = {
    "anime": {
        "handle": "splcher/animefacedataset",
        "dest": "data/anime_faces",
        "description": "Anime Face Dataset (~43k images)",
    },
    "celeba": {
        "handle": "jessicali9530/celeba-dataset",
        "dest": "data/celeba",
        "description": "CelebA Dataset (~202k images)",
    },
}


def _collect_images(src_root: str):
    """Return all image paths found recursively under src_root."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(src_root, "**", ext), recursive=True))
    return paths


def download(dataset_key: str = "anime") -> str:
    """
    Download a Kaggle dataset via kagglehub and copy images to the target
    directory under the project root.

    Args:
        dataset_key: one of {'anime', 'celeba'}

    Returns:
        Absolute path to the destination directory.
    """
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install kagglehub"
        )

    cfg = DATASETS[dataset_key]
    handle = cfg["handle"]
    dest = cfg["dest"]
    description = cfg["description"]

    print(f"Downloading {description}  ({handle}) ...")
    cache_path = kagglehub.dataset_download(handle)
    print(f"Cached at: {cache_path}")

    # Make sure destination directory exists
    os.makedirs(dest, exist_ok=True)

    # Check if already populated
    existing = _collect_images(dest)
    if existing:
        print(f"Destination already contains {len(existing)} images: {dest}")
        print("Skipping copy. Delete the directory to re-download.")
        return os.path.abspath(dest)

    # Collect all images from the cache and copy flat into dest
    images = _collect_images(cache_path)
    if not images:
        raise RuntimeError(
            f"No images found in cache path: {cache_path}\n"
            "Please check your Kaggle credentials and dataset handle."
        )

    print(f"Copying {len(images)} images to {dest} ...")
    for i, src in enumerate(images):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(dest, f"{i:06d}{ext}")
        shutil.copy2(src, dst)
        if (i + 1) % 5000 == 0:
            print(f"  Copied {i + 1}/{len(images)} ...")

    print(f"Done. {len(images)} images available at: {dest}")
    return os.path.abspath(dest)


def main():
    parser = argparse.ArgumentParser(description="Download GAN training datasets via kagglehub")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="anime",
        help="Dataset to download (default: anime)",
    )
    args = parser.parse_args()
    download(args.dataset)


if __name__ == "__main__":
    main()
