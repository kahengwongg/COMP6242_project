"""
Model layer modules.
Contains Self-Attention and other shared components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-Attention module.
    Captures long-range dependencies in feature maps.

    Reference: SAGAN (Self-Attention Generative Adversarial Networks)
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        # 1x1 convolutions for Query, Key, Value
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling parameter gamma
        self.gamma = nn.Parameter(torch.zeros(1))

        # Softmax for computing attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input feature map, shape [B, C, H, W]

        Returns:
            output feature map, shape [B, C, H, W]
        """
        batch_size, C, H, W = x.size()

        # Query: [B, C//8, H, W] -> [B, H*W, C//8]
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)

        # Key: [B, C//8, H, W] -> [B, C//8, H*W]
        key = self.key_conv(x).view(batch_size, -1, H * W)

        # Value: [B, C, H, W] -> [B, C, H*W]
        value = self.value_conv(x).view(batch_size, -1, H * W)

        # Attention weights: [B, H*W, H*W]
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        # Apply attention to Value: [B, H*W, C]
        out = torch.bmm(value, attention.permute(0, 2, 1))

        # Reshape back to original shape: [B, C, H, W]
        out = out.view(batch_size, C, H, W)

        # Residual connection
        out = self.gamma * out + x

        return out


class ConvBlock(nn.Module):
    """
    Basic convolution block.
    Used in Discriminator.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 use_bn=True, use_leaky=True):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if use_leaky:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LayerNormConvBlock(nn.Module):
    """
    Convolution block with LayerNorm instead of BatchNorm.
    Used in WGAN-GP / Combined discriminators.

    LayerNorm requires explicit spatial dimensions [C, H, W],
    so `output_size` must be provided when `use_ln=True`.
    """

    def __init__(self, in_channels, out_channels, output_size,
                 kernel_size=4, stride=2, padding=1, use_ln=True, use_leaky=True):
        super(LayerNormConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

        if use_ln:
            layers.append(nn.LayerNorm([out_channels, output_size, output_size]))

        if use_leaky:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """
    Transposed convolution block.
    Used in Generator.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 use_bn=True, use_relu=True):
        super(ConvTransposeBlock, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def weights_init(m):
    """
    Weight initialization following DCGAN paper.
    Normal distribution (mean=0, std=0.02) for Conv2d, ConvTranspose2d, Linear.
    BatchNorm2d: weight N(1, 0.02), bias=0.
    """
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias.data, 0)