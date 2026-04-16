# Models module
from .dcgan import DCGANGenerator, DCGANDiscriminator
from .wgan_gp import WGGANGenerator, WGGANDiscriminator, compute_gradient_penalty
from .attention_gan import AttentionGANGenerator, AttentionGANDiscriminator
from .combined import CombinedGenerator, CombinedDiscriminator, compute_gradient_penalty
