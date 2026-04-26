import argparse
import sys
import time

import torch
import torch.nn as nn

from models.dcgan import DCGANGenerator, DCGANDiscriminator
from models.layers import weights_init


def get_device(device_arg="auto"):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def try_batch_size(device, batch_size, img_size, z_dim, channels):
    G = DCGANGenerator(z_dim=z_dim, channels=channels).to(device)
    D = DCGANDiscriminator(channels=channels).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    real_images = torch.randn(batch_size, channels, img_size, img_size, device=device)
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # D step
    d_opt.zero_grad(set_to_none=True)
    d_real = D(real_images)
    d_real_loss = criterion(d_real, real_labels)
    z = torch.randn(batch_size, z_dim, device=device)
    fake = G(z).detach()
    d_fake = D(fake)
    d_fake_loss = criterion(d_fake, fake_labels)
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_opt.step()

    # G step
    g_opt.zero_grad(set_to_none=True)
    z = torch.randn(batch_size, z_dim, device=device)
    fake = G(z)
    g_out = D(fake)
    g_loss = criterion(g_out, real_labels)
    g_loss.backward()
    g_opt.step()


def estimate_max_batch(device, img_size, z_dim, channels, start, max_try, safety_margin):
    torch.backends.cudnn.benchmark = True

    def clear_cache():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps":
            torch.mps.empty_cache()

    ok = start
    bs = start
    while bs <= max_try:
        try:
            clear_cache()
            try_batch_size(device, bs, img_size, z_dim, channels)
            ok = bs
            bs *= 2
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps" in msg:
                break
            raise

    low = ok
    high = min(bs, max_try)

    # Binary search between low (ok) and high (maybe fail)
    while low + 1 < high:
        mid = (low + high) // 2
        try:
            clear_cache()
            try_batch_size(device, mid, img_size, z_dim, channels)
            low = mid
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps" in msg:
                high = mid
            else:
                raise

    safe = int(low * safety_margin)
    safe = max(1, safe)
    return low, safe


def main():
    parser = argparse.ArgumentParser(
        description="Estimate max batch size for DCGAN with a single train step."
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--start", type=int, default=16)
    parser.add_argument("--max_try", type=int, default=4096)
    parser.add_argument("--safety_margin", type=float, default=0.85)

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    t0 = time.time()
    max_bs, safe_bs = estimate_max_batch(
        device,
        args.img_size,
        args.z_dim,
        args.channels,
        args.start,
        args.max_try,
        args.safety_margin,
    )
    dt = time.time() - t0

    print(f"Estimated max batch size (1 step): {max_bs}")
    print(f"Recommended safe batch size: {safe_bs} (safety_margin={args.safety_margin})")
    print(f"Elapsed: {dt:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
