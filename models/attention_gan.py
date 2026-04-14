"""
M3: DCGAN + Self-Attention
- Same architecture as M1 DCGAN, with Self-Attention block inserted after the second-to-last layer in G and D
- Loss: BCE
"""

import torch
import torch.nn as nn
from .layers import ConvBlock, ConvTransposeBlock, SelfAttention, weights_init


class AttentionGANGenerator(nn.Module):
    """
    Self-Attention GAN Generator
    
    Same canonical DCGAN-style backbone with Self-Attention inserted at 32x32 resolution.
    
    Architecture:
        Input: z [B, z_dim] reshaped to [B, z_dim, 1, 1]
        ConvT(1x1->4x4) -> ConvT(4x4->8x8) -> ConvT(8x8->16x16) -> ConvT(16x16->32x32)
        -> SelfAttention -> ConvT(32x32->64x64) -> Tanh
    """
    
    def __init__(self, z_dim=100, channels=3, features_g=64):
        super(AttentionGANGenerator, self).__init__()
        
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
        
        # Self-Attention after second-to-last layer (32x32)
        self.attention = SelfAttention(features_g * 2)
        
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
        
        # Self-Attention
        x = self.attention(x)
        
        x = self.conv4(x)  # [B, channels, 64, 64]
        
        return x


class AttentionGANDiscriminator(nn.Module):
    """
    Self-Attention GAN Discriminator
    
    Architecture:
        Conv -> Conv -> Conv -> SelfAttention -> Conv -> Sigmoid
    """
    
    def __init__(self, channels=3, features_d=64):
        super(AttentionGANDiscriminator, self).__init__()
        
        # 64x64 -> 32x32
        self.conv1 = ConvBlock(channels, features_d, use_bn=False, use_leaky=True)
        # 32x32 -> 16x16
        self.conv2 = ConvBlock(features_d, features_d * 2, use_bn=True, use_leaky=True)
        # 16x16 -> 8x8
        self.conv3 = ConvBlock(features_d * 2, features_d * 4, use_bn=True, use_leaky=True)
        
        # Self-Attention after second-to-last layer (8x8)
        self.attention = SelfAttention(features_d * 4)
        
        # 8x8 -> 4x4
        self.conv4 = ConvBlock(features_d * 4, features_d * 8, use_bn=True, use_leaky=True)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)  # [B, features_d, 32, 32]
        x = self.conv2(x)  # [B, features_d*2, 16, 16]
        x = self.conv3(x)  # [B, features_d*4, 8, 8]
        
        # Self-Attention
        x = self.attention(x)
        
        x = self.conv4(x)  # [B, features_d*8, 4, 4]
        x = self.output(x)  # [B, 1, 1, 1]
        
        return x.view(x.size(0), -1)


def test():
    """Test model"""
    batch_size = 16
    z_dim = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    G = AttentionGANGenerator(z_dim=z_dim).to(device)
    D = AttentionGANDiscriminator().to(device)
    
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
    
    # Print parameter counts
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nGenerator params: {g_params:,}")
    print(f"Discriminator params: {d_params:,}")


if __name__ == '__main__':
    test()