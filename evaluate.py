"""
FID Evaluation Module
Primary path: pytorch-fid library (reliable, standard)
Fallback path: custom Inception feature extraction + scipy FID calculation

Usage:
    python evaluate.py --exp_dir experiments/dcgan_full_data_seed42 --data_dir data/anime_faces
    python evaluate.py --exp_dir experiments/wgan_gp_low_data_seed42 --num_samples 5000
"""

import os
import shutil
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.dcgan import DCGANGenerator
from models.wgan_gp import WGGANGenerator
from models.attention_gan import AttentionGANGenerator
from models.combined import CombinedGenerator
from utils.data_loader import ImageFolderFlat, get_transforms

try:
    from scipy import linalg  # used by custom FID fallback
except ImportError:
    linalg = None


# ---------------------------------------------------------------------------
# Device helper (matches train.py: CUDA > MPS > CPU)
# ---------------------------------------------------------------------------

def get_device(device_arg='auto'):
    """Resolve device string to a torch.device, matching train.py logic."""
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Fixed real evaluation subset
# ---------------------------------------------------------------------------

def prepare_real_eval_images(data_dir, output_dir, num_samples, seed=0):
    """
    Prepare a fixed subset of real images for FID evaluation.
    Copies images once; subsequent calls reuse the cache.

    Args:
        data_dir: root directory with training images
        output_dir: where to store the fixed subset (e.g. data/eval_real_5000)
        num_samples: number of real images to select
        seed: random seed for subset selection (separate from training seed)

    Returns:
        output_dir path
    """
    # Check if already prepared with correct count
    if os.path.isdir(output_dir):
        existing = [f for f in os.listdir(output_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(existing) >= num_samples:
            print(f"Using cached real image subset: {output_dir} ({len(existing)} images)")
            return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Collect all image paths
    import glob
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(sorted(glob.glob(os.path.join(data_dir, '**', ext), recursive=True)))

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {data_dir}")

    # Sample with fixed seed
    rng = np.random.RandomState(seed)
    num_to_sample = min(num_samples, len(image_files))
    indices = rng.choice(len(image_files), num_to_sample, replace=False)

    print(f"Preparing real image evaluation subset: {num_to_sample} images (seed={seed}) -> {output_dir}")
    for i, idx in enumerate(tqdm(indices, desc='Copying real images')):
        src = image_files[idx]
        ext = os.path.splitext(src)[1]
        dst = os.path.join(output_dir, f'{i:05d}{ext}')
        shutil.copy2(src, dst)

    return output_dir


# ---------------------------------------------------------------------------
# pytorch-fid wrapper (primary FID path)
# ---------------------------------------------------------------------------

def calculate_fid_pytorch_fid(real_dir, fake_dir, device, batch_size=64):
    """
    Compute FID using the pytorch-fid library.

    Args:
        real_dir: directory of real images
        fake_dir: directory of generated images
        device: torch device
        batch_size: batch size for Inception feature extraction

    Returns:
        fid_score (float)

    Raises:
        ImportError if pytorch-fid is not installed
    """
    from pytorch_fid import fid_score

    device_str = str(device)
    # pytorch-fid expects 'cuda:0' style or 'cpu'
    # MPS is not natively supported by pytorch-fid; fall back to CPU for feature extraction
    if 'mps' in device_str:
        device_str = 'cpu'

    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device_str,
        dims=2048,
    )
    return float(fid_value)


# ---------------------------------------------------------------------------
# Custom FID fallback (kept for environments where pytorch-fid is unavailable)
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """
    InceptionV3 Feature Extractor (fallback)
    Used for computing FID
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load pretrained InceptionV3
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                 transform_input=False)
        inception.eval()
        
        self.model = inception.to(device)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Extract features.
        
        Args:
            x: input images [B, 3, H, W], range [-1, 1]
        
        Returns:
            features: [B, 2048]
        """
        # Resize images to Inception required size (299x299)
        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1] (from [-1, 1])
        x = (x + 1) / 2
        
        # Inception expected normalization
        x = self.normalize(x)
        
        # Extract features (output before avgpool)
        with torch.no_grad():
            x = self.model.Conv2d_1a_3x3(x)
            x = self.model.Conv2d_2a_3x3(x)
            x = self.model.Conv2d_2b_3x3(x)
            x = self.model.maxpool1(x)
            x = self.model.Conv2d_3b_1x1(x)
            x = self.model.Conv2d_4a_3x3(x)
            x = self.model.maxpool2(x)
            x = self.model.Mixed_5b(x)
            x = self.model.Mixed_5c(x)
            x = self.model.Mixed_5d(x)
            x = self.model.Mixed_6a(x)
            x = self.model.Mixed_6b(x)
            x = self.model.Mixed_6c(x)
            x = self.model.Mixed_6d(x)
            x = self.model.Mixed_6e(x)
            x = self.model.Mixed_7a(x)
            x = self.model.Mixed_7b(x)
            x = self.model.Mixed_7c(x)
            x = self.model.avgpool(x)
            x = self.model.dropout(x)
            x = x.view(x.size(0), -1)  # [B, 2048]
        
        return x


def get_activations(dataloader, model, device='cuda', max_samples=None):
    """
    Get Inception features for a dataset.
    
    Args:
        dataloader: data loader
        model: Inception feature extractor
        device: device
        max_samples: maximum number of samples
    
    Returns:
        activations: numpy array [N, 2048]
    """
    activations = []
    
    for batch in tqdm(dataloader, desc='Extracting features'):
        batch = batch.to(device)
        features = model(batch)
        activations.append(features.cpu().numpy())
        
        if max_samples and len(activations) * batch.size(0) >= max_samples:
            break
    
    activations = np.concatenate(activations, axis=0)
    
    if max_samples:
        activations = activations[:max_samples]
    
    return activations


def calculate_fid(real_activations, fake_activations):
    """
    Compute FID score.
    
    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 * sqrt(Sigma_r * Sigma_f))
    
    Args:
        real_activations: real image features [N, D]
        fake_activations: generated image features [M, D]
    
    Returns:
        fid_score: FID score
    """
    # Compute mean and covariance
    mu_real = np.mean(real_activations, axis=0)
    mu_fake = np.mean(fake_activations, axis=0)
    
    sigma_real = np.cov(real_activations, rowvar=False)
    sigma_fake = np.cov(fake_activations, rowvar=False)
    
    # Ensure covariance matrices are 2D
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
    if sigma_fake.ndim == 0:
        sigma_fake = np.array([[sigma_fake]])
    
    # Compute mean difference
    diff = mu_real - mu_fake
    
    # Compute sqrt(Sigma_r * Sigma_f) using matrix square root
    # Use scipy.linalg.sqrtm for matrix square root
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    # Handle numerical issues (may get complex numbers)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset))
    
    # Ensure result is real
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Compute FID
    fid = diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    
    return float(fid)


def generate_samples(generator, num_samples, z_dim, batch_size, device, save_dir=None):
    """
    Generate samples.
    
    Args:
        generator: generator model
        num_samples: number of samples to generate
        z_dim: latent vector dimension
        batch_size: batch size
        device: device
        save_dir: save directory (optional)
    
    Returns:
        samples: generated image tensors
    """
    generator.eval()
    samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc='Generating samples'):
            # Last batch may have fewer than batch_size
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            z = torch.randn(current_batch_size, z_dim, device=device)
            fake_images = generator(z)
            samples.append(fake_images.cpu())
            
            # Save images
            if save_dir:
                for j, img in enumerate(fake_images):
                    idx = i * batch_size + j
                    if idx >= num_samples:
                        break
                    save_image(img, os.path.join(save_dir, f'{idx:05d}.png'), 
                              normalize=True, value_range=(-1, 1))
    
    generator.train()
    
    return torch.cat(samples, dim=0)


def load_generator(model_name, checkpoint_path, z_dim=100, device='cuda'):
    """
    Load generator model.
    
    Args:
        model_name: model name
        checkpoint_path: checkpoint path
        z_dim: latent vector dimension
        device: device
    
    Returns:
        generator: loaded generator
    """
    model_name = model_name.lower()
    
    if model_name == 'dcgan':
        G = DCGANGenerator(z_dim=z_dim)
    elif model_name == 'wgan_gp':
        G = WGGANGenerator(z_dim=z_dim)
    elif model_name == 'attention_gan':
        G = AttentionGANGenerator(z_dim=z_dim)
    elif model_name == 'combined':
        G = CombinedGenerator(z_dim=z_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'G_state_dict' in checkpoint:
        G.load_state_dict(checkpoint['G_state_dict'])
    else:
        G.load_state_dict(checkpoint)
    
    return G.to(device)


def evaluate(args):
    """
    Main evaluation function.
    Primary: pytorch-fid library.  Fallback: custom Inception + scipy.
    """
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load config
    config_path = os.path.join(args.exp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config.get('model', 'dcgan')
        z_dim = config.get('z_dim', 100)
        condition = config.get('condition', 'full_data')
    else:
        # Infer from directory name, e.g. "attention_gan_low_data_seed42"
        exp_name = os.path.basename(args.exp_dir)
        known_models = ['attention_gan', 'wgan_gp', 'combined', 'dcgan']
        model_name = 'dcgan'
        for m in known_models:
            if exp_name.startswith(m):
                model_name = m
                break
        z_dim = 100
        condition = 'full_data'
    
    print(f"Model: {model_name}, Condition: {condition}")
    
    # Load generator
    checkpoint_path = os.path.join(args.exp_dir, 'generator_final.pt')
    if not os.path.exists(checkpoint_path):
        # Try loading from checkpoints directory
        checkpoints_dir = os.path.join(args.exp_dir, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')])
            if checkpoints:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
    
    print(f"Loading model: {checkpoint_path}")
    G = load_generator(model_name, checkpoint_path, z_dim, device)
    
    # ---- Prepare real images (fixed subset for fair comparison) ----
    real_eval_dir = os.path.join(
        os.path.dirname(args.data_dir.rstrip('/')),
        f'eval_real_{args.num_samples}'
    )
    prepare_real_eval_images(args.data_dir, real_eval_dir, args.num_samples, seed=0)
    
    # ---- Generate fake images to folder ----
    # Seed the generator for reproducible FID samples
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    print(f"\nGenerating {args.num_samples} samples...")
    fake_samples_dir = os.path.join(args.exp_dir, 'fid_samples')
    # Clear previous samples to ensure exact count
    if os.path.exists(fake_samples_dir):
        shutil.rmtree(fake_samples_dir)
    os.makedirs(fake_samples_dir, exist_ok=True)
    
    generate_samples(G, args.num_samples, z_dim, args.batch_size,
                     device, save_dir=fake_samples_dir)
    
    # ---- Compute FID ----
    fid_value = None
    fid_method = None

    # Primary: pytorch-fid
    try:
        print("\nComputing FID (pytorch-fid)...")
        fid_value = calculate_fid_pytorch_fid(real_eval_dir, fake_samples_dir,
                                               device, args.batch_size)
        fid_method = 'pytorch-fid'
    except ImportError:
        print("pytorch-fid not installed, using fallback custom implementation...")
        print("  Hint: pip install pytorch-fid")

    # Fallback: custom implementation
    if fid_value is None:
        feature_extractor = InceptionFeatureExtractor(device)

        transform = get_transforms(img_size=64, augment=False)
        real_dataset = ImageFolderFlat(real_eval_dir, transform=transform)
        real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)

        # Load generated samples as tensors for fallback path
        fake_dataset = ImageFolderFlat(fake_samples_dir, transform=transform)
        fake_dataloader = DataLoader(fake_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)

        print("\nExtracting real image features (fallback)...")
        real_activations = get_activations(real_dataloader, feature_extractor,
                                            device, args.num_samples)

        print("Extracting generated image features (fallback)...")
        fake_activations = get_activations(fake_dataloader, feature_extractor,
                                            device, args.num_samples)

        print("\nComputing FID (fallback)...")
        fid_value = calculate_fid(real_activations, fake_activations)
        fid_method = 'custom-fallback'

    print(f"\n{'='*50}")
    print(f"FID Score: {fid_value:.2f}  (method: {fid_method})")
    print(f"{'='*50}")
    
    # Save results
    results = {
        'fid_score': fid_value,
        'fid_method': fid_method,
        'num_samples': args.num_samples,
        'model': model_name,
        'condition': condition,
    }
    
    results_path = os.path.join(args.exp_dir, 'fid_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return fid_value


def main():
    parser = argparse.ArgumentParser(description='FID Evaluation Script')
    
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--data_dir', type=str, default='data/anime_faces',
                        help='Real dataset directory')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples for FID computation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/mps/cpu)')
    parser.add_argument('--save_samples', action='store_true',
                        help='Whether to save generated samples (already saved by default for FID)')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()