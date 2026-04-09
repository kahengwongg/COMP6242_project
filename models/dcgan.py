"""
M1: Baseline DCGAN
- Generator: 5 ConvTranspose2d layers (1x1->4x4->8x8->16x16->32x32->64x64) + BatchNorm + ReLU + Tanh
- Discriminator: 4x Conv2d + BatchNorm + LeakyReLU + Sigmoid
- Loss: BCE
"""

import torch
import torch.nn as nn
from .layers import ConvBlock, ConvTransposeBlock, weights_init


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator

    Architecture:
        Input: z (latent vector, dim=100) reshaped to [B, z_dim, 1, 1]
        ConvT(1x1->4x4) -> ConvT(4x4->8x8) -> ConvT(8x8->16x16) -> ConvT(16x16->32x32) -> ConvT(32x32->64x64) -> Tanh
        Output: image [B, 3, 64, 64]
    """

    def __init__(self, z_dim=100, channels=3, features_g=64):
        """
        Args:
            z_dim: dimensionality of the latent vector
            channels: number of output image channels
            features_g: base feature map size
        """
        super(DCGANGenerator, self).__init__()

        self.z_dim = z_dim

        # Input: z [B, z_dim, 1, 1]
        # 1x1 -> 4x4
        self.conv0 = ConvTransposeBlock(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0, use_bn=True,
                                        use_relu=True)
        # 4x4 -> 8x8
        self.conv1 = ConvTransposeBlock(features_g * 16, features_g * 8, use_bn=True, use_relu=True)
        # 8x8 -> 16x16
        self.conv2 = ConvTransposeBlock(features_g * 8, features_g * 4, use_bn=True, use_relu=True)
        # 16x16 -> 32x32
        self.conv3 = ConvTransposeBlock(features_g * 4, features_g * 2, use_bn=True, use_relu=True)
        # 32x32 -> 64x64
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(features_g * 2, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # output range [-1, 1]
        )

    def forward(self, z):
        """
        Args:
            z: latent vector [B, z_dim]

        Returns:
            generated image [B, channels, 64, 64]
        """
        # Reshape z to [B, z_dim, 1, 1]
        x = z.view(z.size(0), -1, 1, 1)

        # Upsample
        x = self.conv0(x)  # [B, features_g*16, 4, 4]
        x = self.conv1(x)  # [B, features_g*8, 8, 8]
        x = self.conv2(x)  # [B, features_g*4, 16, 16]
        x = self.conv3(x)  # [B, features_g*2, 32, 32]
        x = self.conv4(x)  # [B, channels, 64, 64]

        return x


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator

    Architecture:
        Input: image [B, 3, 64, 64]
        Conv -> Conv -> Conv -> Conv -> final Conv -> Sigmoid
        Output: real/fake probability [B, 1]
    """

    def __init__(self, channels=3, features_d=64):
        """
        Args:
            channels: number of input image channels
            features_d: base feature map size
        """
        super(DCGANDiscriminator, self).__init__()

        # 64x64 -> 32x32
        self.conv1 = ConvBlock(channels, features_d, use_bn=False, use_leaky=True)
        # 32x32 -> 16x16
        self.conv2 = ConvBlock(features_d, features_d * 2, use_bn=True, use_leaky=True)
        # 16x16 -> 8x8
        self.conv3 = ConvBlock(features_d * 2, features_d * 4, use_bn=True, use_leaky=True)
        # 8x8 -> 4x4
        self.conv4 = ConvBlock(features_d * 4, features_d * 8, use_bn=True, use_leaky=True)

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: input image [B, channels, 64, 64]

        Returns:
            real/fake probability [B, 1, 1, 1] -> [B, 1]
        """
        x = self.conv1(x)  # [B, features_d, 32, 32]
        x = self.conv2(x)  # [B, features_d*2, 16, 16]
        x = self.conv3(x)  # [B, features_d*4, 8, 8]
        x = self.conv4(x)  # [B, features_d*8, 4, 4]
        x = self.output(x)  # [B, 1, 1, 1]

        return x.view(x.size(0), -1)  # [B, 1]


def test():
    """Test model shapes and parameter counts"""
    batch_size = 16
    z_dim = 100

    # Create models
    G = DCGANGenerator(z_dim=z_dim)
    D = DCGANDiscriminator()

    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)

    # Test Generator
    z = torch.randn(batch_size, z_dim)
    fake_images = G(z)
    print(f"Generator output: {fake_images.shape}")

    # Test Discriminator
    output = D(fake_images)
    print(f"Discriminator output: {output.shape}")

    # Print parameter counts
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")


if __name__ == '__main__':
    test()