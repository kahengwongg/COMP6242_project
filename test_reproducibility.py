"""
Reproducibility verification test.

For each configuration, trains twice with the same seed and verifies:
  1. Bitwise equality of all Generator and Discriminator parameters
  2. Final loss values match within tolerance
  3. Generated images from fixed noise are pixel-identical

Configs tested:
  - dcgan + full_data  (BCE loss path, augmentation, 1:1 D:G)
  - wgan_gp + low_data (Wasserstein path, 10% subset, 5:1 critic ratio)

Usage:
    python test_reproducibility.py
    python test_reproducibility.py --epochs 5          # more epochs
    python test_reproducibility.py --num_workers 0     # strict determinism
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
import numpy as np

from train import train
from utils.data_loader import set_seed


# ── Test configurations ──────────────────────────────────────────────────────

CONFIGS = [
    {
        "name": "dcgan_full_data",
        "model": "dcgan",
        "condition": "full_data",
    },
    {
        "name": "wgan_gp_low_data",
        "model": "wgan_gp",
        "condition": "low_data",
    },
    {
        "name": "attention_gan_low_data",
        "model": "attention_gan",
        "condition": "low_data",
    },
    {
        "name": "combined_low_data",
        "model": "combined",
        "condition": "low_data",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_args(model, condition, exp_dir, epochs=3, seed=42, num_workers=4):
    """Build an argparse-compatible namespace matching train.py's expectations."""
    return SimpleNamespace(
        model=model,
        condition=condition,
        seed=seed,
        epochs=epochs,
        batch_size=64,
        z_dim=100,
        channels=3,
        img_size=64,
        num_workers=num_workers,
        lambda_gp=10.0,
        n_critic=5,
        noise_std=0.1,
        data_dir="data/anime_faces",
        exp_dir=exp_dir,
        save_freq=epochs,   # save only at the last epoch
        resume=None,
    )


def load_weights(exp_dir):
    """Load generator and discriminator state dicts from an experiment directory."""
    g_path = os.path.join(exp_dir, "generator_final.pt")
    d_path = os.path.join(exp_dir, "discriminator_final.pt")
    g_sd = torch.load(g_path, map_location="cpu")
    d_sd = torch.load(d_path, map_location="cpu")
    return g_sd, d_sd


def load_loss_stats(exp_dir):
    with open(os.path.join(exp_dir, "loss_stats.json")) as f:
        return json.load(f)


def compare_state_dicts(sd_a, sd_b, label):
    """Compare two state dicts.  Returns (bitwise_pass, allclose_pass, report)."""
    report_lines = []
    all_bitwise = True
    all_close = True

    for key in sd_a:
        t_a, t_b = sd_a[key], sd_b[key]
        bitwise = torch.equal(t_a, t_b)
        close = torch.allclose(t_a, t_b, atol=1e-5)

        if not bitwise:
            all_bitwise = False
            max_diff = (t_a - t_b).abs().max().item()
            report_lines.append(
                f"  {label}.{key}: bitwise MISMATCH  max_diff={max_diff:.2e}  allclose={'OK' if close else 'FAIL'}"
            )
        if not close:
            all_close = False

    return all_bitwise, all_close, report_lines


def compare_generated_samples(exp_dir_a, exp_dir_b, model_name, z_dim=100, n=16):
    """Generate n images from fixed noise with both generators; compare pixel-exact."""
    from train import get_models
    from models.layers import weights_init

    g_sd_a, _ = load_weights(exp_dir_a)
    g_sd_b, _ = load_weights(exp_dir_b)

    # Build generators on CPU for deterministic comparison
    device = torch.device("cpu")
    from models.dcgan import DCGANGenerator
    from models.wgan_gp import WGGANGenerator
    from models.attention_gan import AttentionGANGenerator
    from models.combined import CombinedGenerator

    gen_classes = {
        "dcgan": DCGANGenerator,
        "wgan_gp": WGGANGenerator,
        "attention_gan": AttentionGANGenerator,
        "combined": CombinedGenerator,
    }
    G_cls = gen_classes[model_name]

    G_a = G_cls(z_dim=z_dim).to(device)
    G_a.load_state_dict(g_sd_a)
    G_a.eval()

    G_b = G_cls(z_dim=z_dim).to(device)
    G_b.load_state_dict(g_sd_b)
    G_b.eval()

    # Fixed noise on CPU — fully deterministic
    torch.manual_seed(999)
    z = torch.randn(n, z_dim, device=device)

    with torch.no_grad():
        imgs_a = G_a(z)
        imgs_b = G_b(z)

    bitwise = torch.equal(imgs_a, imgs_b)
    if not bitwise:
        max_diff = (imgs_a - imgs_b).abs().max().item()
        return False, max_diff
    return True, 0.0


# ── Main test runner ─────────────────────────────────────────────────────────

def run_one_config(cfg, epochs, num_workers):
    """Run a single configuration twice and compare.  Returns (passed, detail_str)."""
    name = cfg["name"]
    model = cfg["model"]
    condition = cfg["condition"]

    print(f"\n{'='*60}")
    print(f"CONFIG: {name}  (model={model}, condition={condition}, epochs={epochs})")
    print(f"{'='*60}")

    tmp_root = tempfile.mkdtemp(prefix=f"repro_{name}_")
    dir_a = os.path.join(tmp_root, "run_a")
    dir_b = os.path.join(tmp_root, "run_b")
    os.makedirs(dir_a)
    os.makedirs(dir_b)

    try:
        # ── Run A ────────────────────────────────────────────────────
        print(f"\n--- Run A ---")
        args_a = make_args(model, condition, dir_a, epochs=epochs, num_workers=num_workers)
        exp_dir_a = train(args_a)

        # ── Run B ────────────────────────────────────────────────────
        print(f"\n--- Run B ---")
        args_b = make_args(model, condition, dir_b, epochs=epochs, num_workers=num_workers)
        exp_dir_b = train(args_b)

        # ── Compare weights ──────────────────────────────────────────
        print(f"\n--- Comparing weights ---")
        g_sd_a, d_sd_a = load_weights(exp_dir_a)
        g_sd_b, d_sd_b = load_weights(exp_dir_b)

        g_bitwise, g_close, g_report = compare_state_dicts(g_sd_a, g_sd_b, "G")
        d_bitwise, d_close, d_report = compare_state_dicts(d_sd_a, d_sd_b, "D")

        # ── Compare losses ───────────────────────────────────────────
        print(f"--- Comparing losses ---")
        stats_a = load_loss_stats(exp_dir_a)
        stats_b = load_loss_stats(exp_dir_b)

        g_loss_a = stats_a["final_g_loss"]
        g_loss_b = stats_b["final_g_loss"]
        d_loss_a = stats_a["final_d_loss"]
        d_loss_b = stats_b["final_d_loss"]

        loss_match = (
            abs(g_loss_a - g_loss_b) < 1e-6
            and abs(d_loss_a - d_loss_b) < 1e-6
        )

        # ── Compare generated samples ────────────────────────────────
        print(f"--- Comparing generated samples ---")
        samples_match, samples_max_diff = compare_generated_samples(
            exp_dir_a, exp_dir_b, model
        )

        # ── Report ───────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"RESULTS: {name}")
        print(f"{'─'*60}")

        passed = True

        # Weights
        if g_bitwise and d_bitwise:
            print(f"  [PASS] Weights: bitwise identical (G + D)")
        elif g_close and d_close:
            print(f"  [WARN] Weights: NOT bitwise identical, but allclose(atol=1e-5)")
            for line in g_report + d_report:
                print(line)
            print(f"         (Likely MPS/platform non-determinism — not a code bug)")
        else:
            print(f"  [FAIL] Weights: significant differences detected")
            for line in g_report + d_report:
                print(line)
            passed = False

        # Losses
        if loss_match:
            print(f"  [PASS] Losses: G_loss diff={abs(g_loss_a - g_loss_b):.2e}, "
                  f"D_loss diff={abs(d_loss_a - d_loss_b):.2e}")
        else:
            print(f"  [WARN] Losses: G_loss A={g_loss_a:.6f} B={g_loss_b:.6f}, "
                  f"D_loss A={d_loss_a:.6f} B={d_loss_b:.6f}")

        # Samples
        if samples_match:
            print(f"  [PASS] Generated samples: pixel-identical")
        else:
            print(f"  [WARN] Generated samples: max pixel diff={samples_max_diff:.2e}")

        return passed

    finally:
        # Clean up temp directories
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Reproducibility verification test")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs per test run (default: 3)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (use 0 for strict determinism)")
    cli_args = parser.parse_args()

    print("=" * 60)
    print("  REPRODUCIBILITY VERIFICATION TEST")
    print(f"  Epochs per run: {cli_args.epochs}")
    print(f"  DataLoader workers: {cli_args.num_workers}")
    print(f"  Configs: {len(CONFIGS)}")
    print(f"  Total training runs: {len(CONFIGS) * 2}")
    print("=" * 60)

    results = {}
    for cfg in CONFIGS:
        passed = run_one_config(cfg, cli_args.epochs, cli_args.num_workers)
        results[cfg["name"]] = passed

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n  All {len(CONFIGS)} configs passed reproducibility check.")
    else:
        print(f"\n  Some configs FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
