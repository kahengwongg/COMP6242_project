"""
Run all 12 GAN experiments sequentially with automatic resume on interruption.

Progress is tracked in experiments/all_progress.json.
Re-running this script skips finished experiments and resumes interrupted ones.

Usage:
    python run_all_auto.py
    python run_all_auto.py --batch_size 2048 --max_epochs 5000 --patience 3
"""

import argparse
import json
import os
import subprocess
import sys

MODELS = ["dcgan", "wgan_gp", "attention_gan", "combined"]
CONDITIONS = ["full_data", "low_data", "noisy"]


def exp_key(model, condition):
    return f"{model}__{condition}"


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


def build_exp_dir(exp_dir, model, condition, data_dir, seed):
    dataset_tag = os.path.basename(os.path.normpath(data_dir))
    exp_name = f"{model}_{dataset_tag}_{condition}_seed{seed}"
    return os.path.join(exp_dir, exp_name)


def exp_already_done(exp_dir, model, condition, data_dir, seed, patience=3):
    """
    Consider an experiment done if ANY of the following is true:
    1. exp_state.json exists and done=True  (new runs)
    2. eval_progress.csv last row has no_improve >= patience  (old/interrupted runs)
    """
    d = build_exp_dir(exp_dir, model, condition, data_dir, seed)

    # Check 1: state file written by new auto_train_eval.py
    state_path = os.path.join(d, "exp_state.json")
    if os.path.isfile(state_path):
        with open(state_path) as f:
            if json.load(f).get("done", False):
                return True

    # Check 2: eval_progress.csv last row
    csv_path = os.path.join(d, "eval_progress.csv")
    if os.path.isfile(csv_path):
        import csv as _csv
        last_row = None
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                last_row = row
        if last_row is not None:
            try:
                if int(last_row["no_improve"]) >= patience:
                    return True
            except (KeyError, ValueError):
                pass

    return False


def run_experiment(args, model, condition):
    cmd = [
        sys.executable,
        "auto_train_eval.py",
        "--model", model,
        "--condition", condition,
        "--data_dir", args.data_dir,
        "--exp_dir", args.exp_dir,
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--img_size", str(args.img_size),
        "--z_dim", str(args.z_dim),
        "--max_epochs", str(args.max_epochs),
        "--eval_every", str(args.eval_every),
        "--save_freq", str(args.save_freq),
        "--num_samples", str(args.num_samples),
        "--patience", str(args.patience),
        "--min_delta", str(args.min_delta),
    ]
    if args.skip_diversity_metrics:
        cmd.append("--skip_diversity_metrics")
    if args.skip_mifid:
        cmd.append("--skip_mifid")
    if args.skip_extra_metrics:
        cmd.append("--skip_extra_metrics")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run all 12 GAN experiments with automatic resume."
    )
    parser.add_argument("--data_dir", type=str, default="data/anime_faces")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--skip_diversity_metrics", action="store_true")
    parser.add_argument("--skip_mifid", action="store_true")
    parser.add_argument("--skip_extra_metrics", action="store_true")
    # Run only specific models or conditions
    parser.add_argument("--models", nargs="+", default=MODELS,
                        choices=MODELS, help="Subset of models to run")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        choices=CONDITIONS, help="Subset of conditions to run")
    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    progress_path = os.path.join(args.exp_dir, "all_progress.json")
    progress = load_progress(progress_path)

    experiments = [(m, c) for m in args.models for c in args.conditions]
    total = len(experiments)

    print(f"Total experiments: {total}", flush=True)
    print(f"Progress file: {progress_path}\n", flush=True)

    for idx, (model, condition) in enumerate(experiments, 1):
        key = exp_key(model, condition)
        header = f"[{idx}/{total}] {model} / {condition}"

        # Check both the progress tracker and the per-experiment state file
        if progress.get(key) == "done" or exp_already_done(
            args.exp_dir, model, condition, args.data_dir, args.seed, args.patience
        ):
            print(f"{header} — already done, skipping.", flush=True)
            progress[key] = "done"
            save_progress(progress_path, progress)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"{header} — starting", flush=True)
        print(f"{'='*60}\n", flush=True)

        progress[key] = "in_progress"
        save_progress(progress_path, progress)

        try:
            run_experiment(args, model, condition)
            progress[key] = "done"
            save_progress(progress_path, progress)
            print(f"\n{header} — DONE\n", flush=True)
        except subprocess.CalledProcessError as e:
            progress[key] = "failed"
            save_progress(progress_path, progress)
            print(f"\n{header} — FAILED (exit code {e.returncode})", flush=True)
            print("Stopping. Re-run to resume from this experiment.", flush=True)
            sys.exit(1)

    print("\nAll experiments complete!", flush=True)


if __name__ == "__main__":
    main()
