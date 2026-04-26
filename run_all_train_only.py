"""
Run all 12 GAN experiments (train only, no eval).
Results go into experiments/anime_train_100ep/.
Automatically resumes if interrupted.

Usage:
    python run_all_train_only.py
    python run_all_train_only.py --epochs 100 --save_freq 20 --batch_size 2048
"""

import argparse
import json
import os
import re
import subprocess
import sys

MODELS = ["dcgan", "wgan_gp", "attention_gan", "combined"]
CONDITIONS = ["full_data", "low_data", "noisy"]


def parse_checkpoint_epoch(filename):
    m = re.search(r"checkpoint_epoch_(\d+)\.pt$", filename)
    return int(m.group(1)) if m else None


def get_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None, 0
    candidates = []
    for name in os.listdir(checkpoint_dir):
        ep = parse_checkpoint_epoch(name)
        if ep is not None:
            candidates.append((ep, os.path.join(checkpoint_dir, name)))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], candidates[-1][0]


def is_done(checkpoint_dir, epochs):
    """True if the final checkpoint (epoch == epochs) already exists."""
    if not os.path.isdir(checkpoint_dir):
        return False
    for name in os.listdir(checkpoint_dir):
        ep = parse_checkpoint_epoch(name)
        if ep == epochs:
            return True
    # Also accept generator_final.pt as completion marker
    exp_dir = os.path.dirname(checkpoint_dir)
    return os.path.isfile(os.path.join(exp_dir, "generator_final.pt"))


def load_progress(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_progress(path, progress):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train all 12 GAN experiments (no eval) with auto-resume."
    )
    parser.add_argument("--data_dir", type=str, default="data/anime_faces")
    parser.add_argument("--exp_dir", type=str, default="experiments/anime_train_100ep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--attn_batch_size", type=int, default=512,
                        help="Batch size for attention_gan/combined (more memory hungry)")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    progress_path = os.path.join(args.exp_dir, "train_progress.json")
    progress = load_progress(progress_path)

    experiments = [(m, c) for m in args.models for c in args.conditions]
    total = len(experiments)
    print(f"Total experiments: {total}", flush=True)
    print(f"Exp dir: {args.exp_dir}", flush=True)
    print(f"Progress file: {progress_path}\n", flush=True)

    for idx, (model, condition) in enumerate(experiments, 1):
        key = f"{model}__{condition}"
        dataset_tag = os.path.basename(os.path.normpath(args.data_dir))
        exp_name = f"{model}_{dataset_tag}_{condition}_seed{args.seed}"
        exp_subdir = os.path.join(args.exp_dir, exp_name)
        checkpoint_dir = os.path.join(exp_subdir, "checkpoints")
        header = f"[{idx}/{total}] {model} / {condition}"

        # Skip if finished
        if progress.get(key) == "done" or is_done(checkpoint_dir, args.epochs):
            print(f"{header} — already done, skipping.", flush=True)
            progress[key] = "done"
            save_progress(progress_path, progress)
            continue

        # Auto-resume from latest checkpoint
        resume_path, last_epoch = get_latest_checkpoint(checkpoint_dir)

        print(f"\n{'='*60}", flush=True)
        print(f"{header} — starting from epoch {last_epoch}", flush=True)
        print(f"{'='*60}\n", flush=True)

        progress[key] = "in_progress"
        save_progress(progress_path, progress)

        # attention_gan/combined need more memory
        effective_bs = args.attn_batch_size if model in ("attention_gan", "combined") else args.batch_size

        cmd = [
            sys.executable, "train.py",
            "--model", model,
            "--condition", condition,
            "--data_dir", args.data_dir,
            "--exp_dir", args.exp_dir,
            "--seed", str(args.seed),
            "--batch_size", str(effective_bs),
            "--img_size", str(args.img_size),
            "--z_dim", str(args.z_dim),
            "--epochs", str(args.epochs),
            "--save_freq", str(args.save_freq),
        ]
        if resume_path:
            cmd += ["--resume", resume_path]

        # Retry with halved batch size on OOM (up to 3 attempts)
        current_bs = effective_bs
        success = False
        for attempt in range(3):
            if attempt > 0:
                current_bs = max(1, current_bs // 2)
                print(f"  Retrying with batch_size={current_bs} (attempt {attempt+1}/3)...", flush=True)
                # Update batch_size in cmd
                bs_idx = cmd.index("--batch_size")
                cmd[bs_idx + 1] = str(current_bs)
                # Refresh resume path after partial run
                resume_path, _ = get_latest_checkpoint(checkpoint_dir)
                if resume_path:
                    if "--resume" in cmd:
                        cmd[cmd.index("--resume") + 1] = resume_path
                    else:
                        cmd += ["--resume", resume_path]

            try:
                subprocess.run(cmd, check=True)
                progress[key] = "done"
                save_progress(progress_path, progress)
                print(f"\n{header} — DONE (batch_size={current_bs})\n", flush=True)
                success = True
                break
            except subprocess.CalledProcessError as e:
                print(f"  {header} — attempt failed (exit {e.returncode})", flush=True)

        if not success:
            progress[key] = "failed"
            save_progress(progress_path, progress)
            print(f"\n{header} — FAILED after retries", flush=True)
            print("Re-run to resume from this experiment.", flush=True)
            sys.exit(1)

    print("\nAll training complete!", flush=True)


if __name__ == "__main__":
    main()
