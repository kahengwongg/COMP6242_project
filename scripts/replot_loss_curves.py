"""
Regenerate loss_curve.png from tensorboard event files.

Motivation
----------
The original utils/visualize.plot_loss_curves() plots g_losses and d_losses
against list index. For WGAN-GP (n_critic=5) the generator list is 5x shorter
than the discriminator list, so the G curve appears to "end early" on the
x-axis even though training ran to completion. See loss_curve.png of any
wgan_gp / combined experiment.

The tensorboard event files (written by train.py via SummaryWriter) store
'Loss/Generator' and 'Loss/Discriminator' under the SAME global_step. So we
can recover a properly-aligned plot without retraining.

Usage
-----
    # replot one experiment
    python scripts/replot_loss_curves.py --exp_dir experiments/wgan_gp_celeba_full_data_seed42

    # replot all 12 experiments
    python scripts/replot_loss_curves.py --all

    # overwrite loss_curve.png (default writes loss_curve_aligned.png)
    python scripts/replot_loss_curves.py --all --overwrite
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard is not installed. Run: pip install tensorboard", file=sys.stderr)
    sys.exit(1)


def load_scalars(event_file_path):
    """
    Read 'Loss/Generator' and 'Loss/Discriminator' scalar series from a
    tensorboard event file.

    Returns:
        g_steps, g_values, d_steps, d_values  (lists of int / float)
    """
    # size_guidance=0 on SCALARS means load all of them (no downsampling)
    ea = EventAccumulator(
        event_file_path,
        size_guidance={'scalars': 0},
    )
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if 'Loss/Generator' not in tags or 'Loss/Discriminator' not in tags:
        raise RuntimeError(
            f"Expected 'Loss/Generator' and 'Loss/Discriminator' tags in "
            f"{event_file_path}, found: {tags}"
        )

    g_events = ea.Scalars('Loss/Generator')
    d_events = ea.Scalars('Loss/Discriminator')

    g_steps = [e.step for e in g_events]
    g_values = [e.value for e in g_events]
    d_steps = [e.step for e in d_events]
    d_values = [e.value for e in d_events]

    return g_steps, g_values, d_steps, d_values


def find_event_file(exp_dir):
    """Return the path of the first tensorboard event file found under exp_dir/logs/."""
    pattern = os.path.join(exp_dir, 'logs', 'events.out.tfevents.*')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No tensorboard event file under {exp_dir}/logs/")
    if len(matches) > 1:
        print(f"  [info] multiple event files under {exp_dir}/logs/, using {matches[-1]}")
    return matches[-1]


def replot(exp_dir, overwrite=False):
    """Regenerate the loss curve from the tensorboard event file."""
    event_file = find_event_file(exp_dir)
    g_steps, g_values, d_steps, d_values = load_scalars(event_file)

    title = os.path.basename(os.path.normpath(exp_dir))

    plt.figure(figsize=(10, 6))
    plt.plot(d_steps, d_values, label='Discriminator Loss', alpha=0.8, color='tab:orange')
    plt.plot(g_steps, g_values, label='Generator Loss', alpha=0.8, color='tab:blue')
    plt.title(title)
    plt.xlabel('Iteration (batch step)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_name = 'loss_curve.png' if overwrite else 'loss_curve_aligned.png'
    out_path = os.path.join(exp_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    ratio = (len(d_values) / max(len(g_values), 1))
    print(f"[ok] {title}")
    print(f"     G: {len(g_values):>7d} points, steps {g_steps[0] if g_steps else '-'}..{g_steps[-1] if g_steps else '-'}")
    print(f"     D: {len(d_values):>7d} points, steps {d_steps[0] if d_steps else '-'}..{d_steps[-1] if d_steps else '-'}")
    print(f"     D/G length ratio = {ratio:.2f} (expected ~5 for WGAN-GP / Combined, ~1 for DCGAN / AttentionGAN)")
    print(f"     -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Single experiment directory (mutually exclusive with --all)')
    parser.add_argument('--all', action='store_true',
                        help='Replot every experiment under experiments/')
    parser.add_argument('--experiments_root', type=str, default='experiments',
                        help='Root folder containing experiment subfolders (for --all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite loss_curve.png instead of writing loss_curve_aligned.png')
    args = parser.parse_args()

    if bool(args.exp_dir) == bool(args.all):
        parser.error("Pass exactly one of --exp_dir or --all")

    if args.exp_dir:
        replot(args.exp_dir, overwrite=args.overwrite)
        return

    # --all mode
    exp_dirs = sorted([
        d for d in glob.glob(os.path.join(args.experiments_root, '*'))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, 'logs'))
    ])
    if not exp_dirs:
        print(f"No experiments with logs/ found under {args.experiments_root}")
        sys.exit(1)

    print(f"Replotting {len(exp_dirs)} experiments...\n")
    for d in exp_dirs:
        try:
            replot(d, overwrite=args.overwrite)
        except Exception as e:
            print(f"[fail] {d}: {e}")
        print()


if __name__ == '__main__':
    main()
