"""
Download datasets for GAN training.

Usage:
    python -m utils.download_data                  # default: anime_faces
    python -m utils.download_data --dataset celeba
"""

import argparse
import shutil
from pathlib import Path

import kagglehub

DATA_ROOT = Path(__file__).parent.parent / "data"

DATASETS = {
    "anime_faces": {
        "kaggle_id": "soumikrakshit/anime-faces",
        "dest": DATA_ROOT / "anime_faces",
    },
    "celeba": {
        "kaggle_id": "jessicali9530/celeba-dataset",
        "dest": DATA_ROOT / "celeba",
    },
}


def download(name: str) -> None:
    info = DATASETS[name]
    dest = info["dest"]

    if dest.exists() and any(dest.rglob("*.jpg")) or any(dest.rglob("*.png")):
        print(f"Dataset already exists at: {dest}")
        return

    path = kagglehub.dataset_download(info["kaggle_id"])
    print(f"Downloaded to cache: {path}")
    shutil.copytree(path, dest, dirs_exist_ok=True)

    count = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png")) + sum(1 for _ in dest.rglob("*.jpeg"))
    print(f"Dataset ready at: {dest}")
    print(f"Images available: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="anime_faces",
        help="Dataset to download (default: anime_faces)",
    )
    args = parser.parse_args()
    download(args.dataset)