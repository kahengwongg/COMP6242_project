"""
M3 Extension: AttentionGAN with BoundedSelfAttention

Two variants for the bounded-attention ablation:
  - BoundedGGenerator / BoundedGDiscriminator  (G-only bounded: only G uses BoundedSelfAttention)
  - BoundedGDGenerator / BoundedGDDiscriminator (G+D bounded: both sides use BoundedSelfAttention)

Original attention_gan.py is untouched; Combined baseline is unaffected.
"""

import torch
import torch.nn as nn
from .layers import ConvBlock, ConvTransposeBlock, SelfAttention, BoundedSelfAttention, weights_init


# ---------------------------------------------------------------------------
# G-only bounded: Generator uses BoundedSelfAttention, Discriminator unchanged
# ---------------------------------------------------------------------------

class BoundedGGenerator(nn.Module):
    def __init__(self, z_dim=100, channels=3, features_g=64, max_gamma=0.5):
        super().__init__()
        self.z_dim = z_dim
        self.conv0 = ConvTransposeBlock(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0,
                                        use_bn=True, use_relu=True)
        self.conv1 = ConvTransposeBlock(features_g * 16, features_g * 8, use_bn=True, use_relu=True)
        self.conv2 = ConvTransposeBlock(features_g * 8, features_g * 4, use_bn=True, use_relu=True)
        self.conv3 = ConvTransposeBlock(features_g * 4, features_g * 2, use_bn=True, use_relu=True)
        self.attention = BoundedSelfAttention(features_g * 2, max_gamma=max_gamma)
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(features_g * 2, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = z.view(z.size(0), -1, 1, 1)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x)
        x = self.conv4(x)
        return x


class BoundedGDiscriminator(nn.Module):
    """Discriminator with original (unbounded) SelfAttention — same as baseline."""
    def __init__(self, channels=3, features_d=64):
        super().__init__()
        self.conv1 = ConvBlock(channels, features_d, use_bn=False, use_leaky=True)
        self.conv2 = ConvBlock(features_d, features_d * 2, use_bn=True, use_leaky=True)
        self.conv3 = ConvBlock(features_d * 2, features_d * 4, use_bn=True, use_leaky=True)
        self.attention = SelfAttention(features_d * 4)
        self.conv4 = ConvBlock(features_d * 4, features_d * 8, use_bn=True, use_leaky=True)
        self.output = nn.Sequential(
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x)
        x = self.conv4(x)
        x = self.output(x)
        return x.view(x.size(0), -1)


# ---------------------------------------------------------------------------
# G+D bounded: both Generator and Discriminator use BoundedSelfAttention
# ---------------------------------------------------------------------------

class BoundedGDGenerator(BoundedGGenerator):
    """Same as BoundedGGenerator — inherits directly."""
    pass


class BoundedGDDiscriminator(nn.Module):
    def __init__(self, channels=3, features_d=64, max_gamma=0.5):
        super().__init__()
        self.conv1 = ConvBlock(channels, features_d, use_bn=False, use_leaky=True)
        self.conv2 = ConvBlock(features_d, features_d * 2, use_bn=True, use_leaky=True)
        self.conv3 = ConvBlock(features_d * 2, features_d * 4, use_bn=True, use_leaky=True)
        self.attention = BoundedSelfAttention(features_d * 4, max_gamma=max_gamma)
        self.conv4 = ConvBlock(features_d * 4, features_d * 8, use_bn=True, use_leaky=True)
        self.output = nn.Sequential(
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x)
        x = self.conv4(x)
        x = self.output(x)
        return x.view(x.size(0), -1)
