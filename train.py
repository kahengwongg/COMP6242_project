"""
Unified training entry point.
Supports 4 models (dcgan, wgan_gp, attention_gan, combined) and 3 conditions (full_data, low_data, noisy)

Usage:
    python train.py --model dcgan --condition full_data --seed 42
    python train.py --model wgan_gp --condition low_data --seed 42
    python train.py --model attention_gan --condition noisy --seed 42
"""

import os
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import models
from models.dcgan import DCGANGenerator, DCGANDiscriminator
from models.wgan_gp import WGGANGenerator, WGGANDiscriminator, compute_gradient_penalty as wgan_gp_compute_gp
from models.attention_gan import AttentionGANGenerator, AttentionGANDiscriminator
from models.combined import CombinedGenerator, CombinedDiscriminator, compute_gradient_penalty as combined_compute_gp
from models.layers import weights_init

# Import utilities
from utils.data_loader import get_dataloader, set_seed
from utils.visualize import save_samples, plot_loss_curves, compute_loss_stats


def get_mem_mb():
    """Return current accelerator memory usage in MB, or None if unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**2
    return None


def get_models(model_name, z_dim=100, channels=3, device='cuda'):
    """
    Create Generator and Discriminator based on model name.
    
    Args:
        model_name: model name ('dcgan', 'wgan_gp', 'attention_gan', 'combined')
        z_dim: latent vector dimension
        channels: image channels
        device: device
    
    Returns:
        (generator, discriminator)
    """
    model_name = model_name.lower()
    
    if model_name == 'dcgan':
        G = DCGANGenerator(z_dim=z_dim, channels=channels)
        D = DCGANDiscriminator(channels=channels)
    elif model_name == 'wgan_gp':
        G = WGGANGenerator(z_dim=z_dim, channels=channels)
        D = WGGANDiscriminator(channels=channels)
    elif model_name == 'attention_gan':
        G = AttentionGANGenerator(z_dim=z_dim, channels=channels)
        D = AttentionGANDiscriminator(channels=channels)
    elif model_name == 'combined':
        G = CombinedGenerator(z_dim=z_dim, channels=channels)
        D = CombinedDiscriminator(channels=channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)
    
    return G.to(device), D.to(device)


def get_optimizers(model_name, G, D, lr_dcgan=2e-4, lr_wgan=1e-4):
    """
    Create optimizers based on model type.
    
    Args:
        model_name: model name
        G: Generator
        D: Discriminator
        lr_dcgan: DCGAN learning rate
        lr_wgan: WGAN-GP learning rate
    
    Returns:
        (g_optimizer, d_optimizer)
    """
    model_name = model_name.lower()
    
    if model_name in ['dcgan', 'attention_gan']:
        # DCGAN uses Adam, β1=0.5 (Radford et al. 2016)
        g_optimizer = optim.Adam(G.parameters(), lr=lr_dcgan, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(D.parameters(), lr=lr_dcgan, betas=(0.5, 0.999))
    else:
        # WGAN-GP uses Adam with β1=0, β2=0.9 (Gulrajani et al. 2017, §4).
        # High first-moment momentum (β1=0.5) can destabilise the critic when
        # combined with gradient penalty; zeroing β1 avoids stale momentum.
        g_optimizer = optim.Adam(G.parameters(), lr=lr_wgan, betas=(0.0, 0.9))
        d_optimizer = optim.Adam(D.parameters(), lr=lr_wgan, betas=(0.0, 0.9))
    
    return g_optimizer, d_optimizer


def train_dcgan_step(G, D, real_images, g_optimizer, d_optimizer, z_dim, device, criterion):
    """
    DCGAN single training step (BCE Loss)
    """
    batch_size = real_images.size(0)
    
    # Train Discriminator
    d_optimizer.zero_grad()
    
    # Real images
    real_labels = torch.ones(batch_size, 1, device=device)
    d_real_output = D(real_images)
    d_real_loss = criterion(d_real_output, real_labels)
    
    # Generated images
    z = torch.randn(batch_size, z_dim, device=device)
    fake_images = G(z).detach()  # detach so D is trained on fake images without backpropagating into G
    fake_labels = torch.zeros(batch_size, 1, device=device)
    d_fake_output = D(fake_images)
    d_fake_loss = criterion(d_fake_output, fake_labels)
    
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()
    
    # Train Generator
    g_optimizer.zero_grad()
    
    z = torch.randn(batch_size, z_dim, device=device)
    fake_images = G(z)  # no detach so gradients from D(fake_images) can backpropagate into G
    g_output = D(fake_images)
    g_loss = criterion(g_output, real_labels)  # Want D to classify generated images as real
    
    g_loss.backward()
    g_optimizer.step()

    d_real_mean = d_real_output.mean().item()
    d_fake_mean = d_fake_output.mean().item()

    return g_loss.item(), d_loss.item(), d_real_mean, d_fake_mean


def train_wgan_gp_step(G, D, real_images, g_optimizer, d_optimizer, z_dim, device, 
                       lambda_gp=10, n_critic=5, compute_gp_func=wgan_gp_compute_gp,
                       critic_step=0):
    """
    WGAN-GP single training step (Wasserstein Loss + Gradient Penalty)
    D:G update ratio = n_critic:1

    Args:
        critic_step: current batch step count; G is updated when critic_step % n_critic == 0
    """
    batch_size = real_images.size(0)
    
    # Train Discriminator (Critic)
    d_optimizer.zero_grad()
    
    # Real images
    d_real = D(real_images)
    
    # Generated images
    z = torch.randn(batch_size, z_dim, device=device)
    fake_images = G(z).detach()
    d_fake = D(fake_images)
    
    # Wasserstein Loss
    d_loss_wasserstein = d_fake.mean() - d_real.mean()
    
    # Gradient Penalty
    gradient_penalty = compute_gp_func(D, real_images, fake_images, device)
    
    d_loss = d_loss_wasserstein + lambda_gp * gradient_penalty
    d_loss.backward()
    d_optimizer.step()
    
    # Train Generator (every n_critic steps)
    g_loss_value = None
    
    if critic_step % n_critic == 0:
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, z_dim, device=device)
        fake_images = G(z)
        g_output = D(fake_images)
        
        g_loss = -g_output.mean()  # Maximize D(fake)
        g_loss.backward()
        g_optimizer.step()
        
        g_loss_value = g_loss.item()

    d_real_mean = d_real.mean().item()
    d_fake_mean = d_fake.mean().item()

    return g_loss_value, d_loss.item(), d_real_mean, d_fake_mean


def train(args):
    """
    Main training function.
    """
    # Set device (auto-select CUDA > MPS > CPU)
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    dataset_tag = os.path.basename(os.path.normpath(args.data_dir))
    exp_name = f"{args.model}_{dataset_tag}_{args.condition}_seed{args.seed}"
    exp_dir = os.path.join(args.exp_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    sample_dir = os.path.join(exp_dir, 'samples')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
    
    # Create data loader
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        condition=args.condition,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        num_workers=args.num_workers,
        noise_std=args.noise_std
    )
    
    # Create models
    G, D = get_models(args.model, args.z_dim, args.channels, device)
    
    # Print model info
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nModel: {args.model}")
    print(f"Generator params: {g_params:,}")
    print(f"Discriminator params: {d_params:,}")
    print(f"Experiment dir: {exp_dir}\n")
    
    # Create optimizers
    g_optimizer, d_optimizer = get_optimizers(args.model, G, D)
    
    # Loss function (only used by DCGAN)
    criterion = nn.BCELoss()
    
    # Fixed noise for generating samples
    fixed_noise = torch.randn(64, args.z_dim, device=device)
    
    # Training records
    g_losses = []
    d_losses = []
    
    # Select training function
    is_wgan = args.model.lower() in ['wgan_gp', 'combined']
    compute_gp_func = wgan_gp_compute_gp if args.model.lower() == 'wgan_gp' else combined_compute_gp
    
    # Resume from checkpoint
    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        G.load_state_dict(checkpoint['G_state_dict'])
        D.load_state_dict(checkpoint['D_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['epoch'] * len(dataloader)
        if 'critic_step' in checkpoint:
            critic_step = checkpoint['critic_step']
        print(f"Resumed from checkpoint: epoch {checkpoint['epoch']} -> continuing from epoch {start_epoch}")
    
    print(f"Starting training: {args.epochs} epochs, {len(dataloader)} batches/epoch")
    start_time = time.time()
    
    # Critic step counter for WGAN-GP (tracks total D updates across epochs)
    if not args.resume:
        critic_step = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(device)
            
            if is_wgan:
                # WGAN-GP training
                critic_step += 1
                g_loss, d_loss, d_real_mean, d_fake_mean = train_wgan_gp_step(
                    G, D, real_images, g_optimizer, d_optimizer,
                    args.z_dim, device, args.lambda_gp, args.n_critic, compute_gp_func,
                    critic_step=critic_step
                )
                
                if g_loss is not None:
                    g_losses.append(g_loss)
                    epoch_g_losses.append(g_loss)
            else:
                # DCGAN training
                g_loss, d_loss, d_real_mean, d_fake_mean = train_dcgan_step(
                    G, D, real_images, g_optimizer, d_optimizer,
                    args.z_dim, device, criterion
                )
                g_losses.append(g_loss)
                epoch_g_losses.append(g_loss)
            
            d_losses.append(d_loss)
            epoch_d_losses.append(d_loss)
            
            # Update progress bar
            mem = get_mem_mb()
            postfix = {}
            if g_loss is not None:
                postfix['G_loss'] = f'{g_loss:.4f}'
            postfix['D_loss'] = f'{d_loss:.4f}'
            postfix['D(real)'] = f'{d_real_mean:.3f}'
            postfix['D(fake)'] = f'{d_fake_mean:.3f}'
            if mem is not None:
                postfix['Mem(MB)'] = f'{mem:.0f}'
            pbar.set_postfix(postfix)
            
            # TensorBoard logging
            if g_loss is not None:
                writer.add_scalar('Loss/Generator', g_loss, global_step)
            writer.add_scalar('Loss/Discriminator', d_loss, global_step)
            writer.add_scalar('D_outputs/real_mean', d_real_mean, global_step)
            writer.add_scalar('D_outputs/fake_mean', d_fake_mean, global_step)
            
            global_step += 1
        
        # Epoch finished
        epoch_time = time.time() - epoch_start_time
        
        # Compute epoch average loss
        if epoch_g_losses:
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        else:
            avg_g_loss = 0
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        
        print(f"Epoch {epoch} done - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}, time: {epoch_time:.1f}s")
        
        # Save samples
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_samples(G, fixed_noise, epoch, sample_dir)
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
                'critic_step': critic_step,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
    
    # Save final models
    torch.save(G.state_dict(), os.path.join(exp_dir, 'generator_final.pt'))
    torch.save(D.state_dict(), os.path.join(exp_dir, 'discriminator_final.pt'))
    
    # Save loss curves
    plot_loss_curves(g_losses, d_losses, os.path.join(exp_dir, 'loss_curve.png'),
                     title=f'{args.model} - {args.condition}')
    
    # Save loss statistics
    loss_stats = {
        'generator': compute_loss_stats(g_losses),
        'discriminator': compute_loss_stats(d_losses),
        'total_time_hours': total_time / 3600,
        'num_epochs': args.epochs,
        'final_g_loss': g_losses[-1] if g_losses else 0,
        'final_d_loss': d_losses[-1]
    }
    
    with open(os.path.join(exp_dir, 'loss_stats.json'), 'w') as f:
        json.dump(loss_stats, f, indent=2)
    
    # Close TensorBoard
    writer.close()
    
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description='GAN Training Script')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='dcgan',
                        choices=['dcgan', 'wgan_gp', 'attention_gan', 'combined'],
                        help='Model type')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Latent vector dimension')
    parser.add_argument('--channels', type=int, default=3,
                        help='Image channels')
    parser.add_argument('--img_size', type=int, default=64,
                        help='Image size')
    
    # Experimental conditions
    parser.add_argument('--condition', type=str, default='full_data',
                        choices=['full_data', 'low_data', 'noisy'],
                        help='Experimental condition')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # WGAN-GP parameters
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Gradient penalty coefficient')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='D:G update ratio')
    
    # Noisy condition parameters
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Noise standard deviation')
    
    # Path parameters
    parser.add_argument('--data_dir', type=str, default='data/anime_faces',
                        help='Dataset directory')
    parser.add_argument('--exp_dir', type=str, default='experiments',
                        help='Experiment results directory')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path')
    
    args = parser.parse_args()
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()