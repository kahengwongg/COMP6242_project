"""
Visualization utilities module.
Contains functions for saving generated samples, plotting loss curves, etc.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


def save_samples(generator, fixed_noise, epoch, save_dir, nrow=8):
    """
    Save generated sample images.
    
    Args:
        generator: generator model
        fixed_noise: fixed noise vector
        epoch: current epoch
        save_dir: save directory
        nrow: number of images per row
    """
    generator.eval()
    
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        
        # Ensure image values are in [-1, 1] range
        fake_images = torch.clamp(fake_images, -1, 1)
        
        # Save as grid image
        save_path = os.path.join(save_dir, f'samples_epoch_{epoch}.png')
        save_image(fake_images, save_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    generator.train()
    
    return save_path


def plot_loss_curves(g_losses, d_losses, save_path, title='Training Loss'):
    """
    Plot loss curves.
    
    Args:
        g_losses: Generator loss list
        d_losses: Discriminator loss list
        save_path: save path
        title: chart title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label='Generator Loss', alpha=0.8)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_multiple_loss_curves(losses_dict, save_path, title='Training Loss Comparison'):
    """
    Plot multiple loss curves comparison chart.
    
    Args:
        losses_dict: dict, key is experiment name, value is (g_losses, d_losses)
        save_path: save path
        title: chart title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Generator Loss
    ax1 = axes[0]
    for name, (g_losses, _) in losses_dict.items():
        ax1.plot(g_losses, label=name, alpha=0.8)
    ax1.set_title('Generator Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Discriminator Loss
    ax2 = axes[1]
    for name, (_, d_losses) in losses_dict.items():
        ax2.plot(d_losses, label=name, alpha=0.8)
    ax2.set_title('Discriminator Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_sample_grid(images, nrow=8, title=None):
    """
    Create sample grid image.
    
    Args:
        images: image tensor [B, C, H, W]
        nrow: images per row
        title: title
    
    Returns:
        matplotlib figure
    """
    # Create grid
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    # Convert to numpy
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14)
    
    return fig


def save_comparison_grid(images_dict, save_path, nrow=4):
    """
    Save multiple sample comparison grids.
    
    Args:
        images_dict: dict, key is experiment name, value is image tensor
        save_path: save path
        nrow: number of rows per sample group
    """
    n_models = len(images_dict)
    n_samples = list(images_dict.values())[0].size(0)
    
    fig, axes = plt.subplots(n_models, 1, figsize=(n_samples * 2, n_models * 3))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, images) in enumerate(images_dict.items()):
        grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        axes[idx].imshow(grid_np)
        axes[idx].axis('off')
        axes[idx].set_title(name, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def denormalize(images):
    """
    Convert normalized images back to [0, 1] range.
    
    Args:
        images: normalized images [-1, 1]
    
    Returns:
        denormalized images [0, 1]
    """
    return (images + 1) / 2


def compute_loss_stats(losses):
    """
    Compute loss statistics.
    
    Args:
        losses: loss list
    
    Returns:
        dict containing mean, std, min, max
    """
    losses_np = np.array(losses)
    
    return {
        'mean': float(np.mean(losses_np)),
        'std': float(np.std(losses_np)),
        'min': float(np.min(losses_np)),
        'max': float(np.max(losses_np))
    }


def plot_fid_comparison(fid_scores, save_path, title='FID Score Comparison'):
    """
    Plot FID score comparison bar chart.
    
    Args:
        fid_scores: dict, key is experiment name, value is FID score
        save_path: save path
        title: chart title
    """
    names = list(fid_scores.keys())
    scores = list(fid_scores.values())
    
    # Sort by score
    sorted_pairs = sorted(zip(names, scores), key=lambda x: x[1])
    names = [p[0] for p in sorted_pairs]
    scores = [p[1] for p in sorted_pairs]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(names)), scores, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{score:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('FID Score (lower is better)')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization module...")
    
    # Create test data
    test_images = torch.randn(64, 3, 64, 64)
    test_g_losses = [np.random.randn() * 2 + 5 for _ in range(100)]
    test_d_losses = [np.random.randn() * 1 + 2 for _ in range(100)]
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    
    # Test loss curves
    plot_loss_curves(test_g_losses, test_d_losses, 'test_output/test_loss.png')
    print("Loss curves saved: test_output/test_loss.png")
    
    # Test statistics
    stats = compute_loss_stats(test_g_losses)
    print(f"Loss statistics: {stats}")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_output')
    print("Test complete!")