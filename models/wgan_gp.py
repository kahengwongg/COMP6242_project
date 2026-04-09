"""
M2: WGAN-GP (Wasserstein GAN with Gradient Penalty)
- Same architecture as M1 DCGAN, but Discriminator removes final Sigmoid
- Loss: Wasserstein distance + Gradient Penalty (λ=10)
- Learning rate: 1e-4
- D:G update ratio = 5:1
"""

import torch
import torch.nn as nn
from .layers import ConvTransposeBlock, LayerNormConvBlock, weights_init


class WGGANGenerator(nn.Module):
    """
    WGAN-GP Generator
    Same architecture as DCGAN Generator (canonical DCGAN-style input)
    
    Architecture:
        Input: z [B, z_dim] reshaped to [B, z_dim, 1, 1]
        ConvT(1x1->4x4) -> ConvT(4x4->8x8) -> ConvT(8x8->16x16) -> ConvT(16x16->32x32) -> ConvT(32x32->64x64) -> Tanh
    """
    
    def __init__(self, z_dim=100, channels=3, features_g=64):
        super(WGGANGenerator, self).__init__()
        
        self.z_dim = z_dim
        
        # Input: z [B, z_dim, 1, 1]
        # 1x1 -> 4x4
        self.conv0 = ConvTransposeBlock(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0, use_bn=True, use_relu=True)
        # 4x4 -> 8x8
        self.conv1 = ConvTransposeBlock(features_g * 16, features_g * 8, use_bn=True, use_relu=True)
        # 8x8 -> 16x16
        self.conv2 = ConvTransposeBlock(features_g * 8, features_g * 4, use_bn=True, use_relu=True)
        # 16x16 -> 32x32
        self.conv3 = ConvTransposeBlock(features_g * 4, features_g * 2, use_bn=True, use_relu=True)
        # 32x32 -> 64x64
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(features_g * 2, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        # Reshape z to [B, z_dim, 1, 1]
        x = z.view(z.size(0), -1, 1, 1)
        
        x = self.conv0(x)  # [B, features_g*16, 4, 4]
        x = self.conv1(x)  # [B, features_g*8, 8, 8]
        x = self.conv2(x)  # [B, features_g*4, 16, 16]
        x = self.conv3(x)  # [B, features_g*2, 32, 32]
        x = self.conv4(x)  # [B, channels, 64, 64]
        return x


class WGGANDiscriminator(nn.Module):
    """
    WGAN-GP Discriminator (Critic)
    Similar to DCGAN Discriminator, but:
    1. Removes final Sigmoid (output is Wasserstein distance estimate)
    2. Does not use BatchNorm (as recommended by WGAN-GP paper)
    """
    
    def __init__(self, channels=3, features_d=64):
        super(WGGANDiscriminator, self).__init__()
        
        # Use LayerNorm instead of BatchNorm
        self.conv1 = LayerNormConvBlock(channels, features_d, 32, use_ln=False)        # 64x64 -> 32x32
        self.conv2 = LayerNormConvBlock(features_d, features_d * 2, 16)                # 32x32 -> 16x16
        self.conv3 = LayerNormConvBlock(features_d * 2, features_d * 4, 8)             # 16x16 -> 8x8
        self.conv4 = LayerNormConvBlock(features_d * 4, features_d * 8, 4)             # 8x8 -> 4x4
        
        # Output layer (no Sigmoid)
        self.output = nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output(x)
        return x.view(x.size(0), -1)


def compute_gradient_penalty(discriminator, real_images, fake_images, device='cuda'):
    """
    Compute Gradient Penalty (GP)
    
    GP = λ * (||∇_x D(x_hat)||_2 - 1)^2
    
    where x_hat is a random interpolation between real and fake images
    
    Args:
        discriminator: discriminator network
        real_images: real images [B, C, H, W]
        fake_images: generated images [B, C, H, W]
        device: device
    
    Returns:
        gradient_penalty: scalar
    """
    batch_size = real_images.size(0)
    
    # Generate random interpolation coefficient epsilon ~ U[0, 1]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Compute x_hat = epsilon * real + (1 - epsilon) * fake
    x_hat = epsilon * real_images + (1 - epsilon) * fake_images
    x_hat.requires_grad_(True)
    
    # Compute D(x_hat)
    d_x_hat = discriminator(x_hat)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_x_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_x_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute L2 norm of gradients
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Compute gradient penalty
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def test():
    """Test model"""
    batch_size = 16
    z_dim = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    G = WGGANGenerator(z_dim=z_dim).to(device)
    D = WGGANDiscriminator().to(device)
    
    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)
    
    # Test Generator
    z = torch.randn(batch_size, z_dim, device=device)
    fake_images = G(z)
    print(f"Generator output: {fake_images.shape}")
    
    # Test Discriminator
    output = D(fake_images)
    print(f"Discriminator output: {output.shape}")
    
    # Test gradient penalty
    real_images = torch.randn(batch_size, 3, 64, 64, device=device)
    gp = compute_gradient_penalty(D, real_images, fake_images, device)
    print(f"Gradient penalty: {gp.item():.4f}")
    
    # Print parameter counts
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nGenerator params: {g_params:,}")
    print(f"Discriminator params: {d_params:,}")


if __name__ == '__main__':
    test()
