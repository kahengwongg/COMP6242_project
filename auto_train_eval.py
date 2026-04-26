import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time


def run_command(cmd, silent=False):
    if not silent:
        print("Running:", " ".join(cmd))
    with open(os.devnull, "w") as devnull:
        out = devnull if silent else None
        subprocess.run(cmd, check=True, stdout=out, stderr=out)


def parse_checkpoint_epoch(filename):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", filename)
    return int(match.group(1)) if match else None


def get_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None, 0
    candidates = []
    for name in os.listdir(checkpoint_dir):
        epoch = parse_checkpoint_epoch(name)
        if epoch is not None:
            candidates.append((epoch, os.path.join(checkpoint_dir, name)))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], candidates[-1][0]


def build_exp_dir(args):
    dataset_tag = os.path.basename(os.path.normpath(args.data_dir))
    exp_name = f"{args.model}_{dataset_tag}_{args.condition}_seed{args.seed}"
    return os.path.join(args.exp_dir, exp_name)


def write_csv_row(path, fieldnames, row):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Train in chunks and evaluate every N epochs with early stopping."
    )
    parser.add_argument("--model", type=str, default="dcgan")
    parser.add_argument("--condition", type=str, default="full_data")
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
    parser.add_argument("--reuse_fake_samples", action="store_true")
    parser.add_argument("--skip_diversity_metrics", action="store_true")
    parser.add_argument("--skip_mifid", action="store_true")
    parser.add_argument("--skip_extra_metrics", action="store_true")

    args = parser.parse_args()

    if args.eval_every <= 0:
        raise ValueError("--eval_every must be > 0")
    if args.max_epochs <= 0:
        raise ValueError("--max_epochs must be > 0")

    exp_dir = build_exp_dir(args)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    results_csv = os.path.join(exp_dir, "eval_progress.csv")
    results_jsonl = os.path.join(exp_dir, "eval_progress.jsonl")
    results_log = os.path.join(exp_dir, "eval_progress.log")
    state_path = os.path.join(exp_dir, "exp_state.json")

    # Restore early-stopping state if interrupted mid-run
    if os.path.isfile(state_path):
        with open(state_path) as f:
            saved = json.load(f)
        if saved.get("done"):
            print(f"[{args.model}/{args.condition}] Already finished, skipping.", flush=True)
            return exp_dir
        best_fid = saved.get("best_fid")
        no_improve = saved.get("no_improve", 0)
        print(
            f"[{args.model}/{args.condition}] Resuming: best_fid={best_fid}, no_improve={no_improve}",
            flush=True,
        )
    else:
        best_fid = None
        no_improve = 0

    resume_path, last_epoch = get_latest_checkpoint(checkpoint_dir)
    start_epoch = last_epoch

    for target_epoch in range(start_epoch + args.eval_every, args.max_epochs + 1, args.eval_every):
        train_cmd = [
            sys.executable,
            "train.py",
            "--model",
            args.model,
            "--condition",
            args.condition,
            "--seed",
            str(args.seed),
            "--epochs",
            str(target_epoch),
            "--batch_size",
            str(args.batch_size),
            "--img_size",
            str(args.img_size),
            "--z_dim",
            str(args.z_dim),
            "--data_dir",
            args.data_dir,
            "--exp_dir",
            args.exp_dir,
            "--save_freq",
            str(args.save_freq),
        ]
        if resume_path:
            train_cmd += ["--resume", resume_path]

        print(f"[Epoch {target_epoch}] Training ...", flush=True)
        run_command(train_cmd, silent=False)

        resume_path, last_epoch = get_latest_checkpoint(checkpoint_dir)
        if last_epoch == 0:
            raise RuntimeError("No checkpoint found after training.")

        # Keep only multiples-of-save_freq + the latest checkpoint
        if os.path.isdir(checkpoint_dir):
            all_eps = []
            for name in os.listdir(checkpoint_dir):
                ep = parse_checkpoint_epoch(name)
                if ep is not None:
                    all_eps.append(ep)
            latest_ep = max(all_eps) if all_eps else 0
            for name in os.listdir(checkpoint_dir):
                ep = parse_checkpoint_epoch(name)
                if ep is not None and ep % args.save_freq != 0 and ep != latest_ep:
                    os.remove(os.path.join(checkpoint_dir, name))

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--exp_dir",
            exp_dir,
            "--data_dir",
            args.data_dir,
            "--num_samples",
            str(args.num_samples),
        ]
        if args.reuse_fake_samples:
            eval_cmd.append("--reuse_fake_samples")
        if args.skip_diversity_metrics:
            eval_cmd.append("--skip_diversity_metrics")
        if args.skip_mifid:
            eval_cmd.append("--skip_mifid")
        if args.skip_extra_metrics:
            eval_cmd.append("--skip_extra_metrics")

        print(f"[Epoch {target_epoch}] Evaluating ...", flush=True)
        eval_start = time.time()
        run_command(eval_cmd, silent=True)
        eval_time = time.time() - eval_start

        fid_path = os.path.join(exp_dir, "fid_results.json")
        if not os.path.isfile(fid_path):
            raise RuntimeError(f"Missing fid_results.json at {fid_path}")

        with open(fid_path, "r") as f:
            fid_data = json.load(f)

        fid_score = fid_data.get("fid_score")
        if fid_score is None:
            raise RuntimeError("fid_results.json missing fid_score")

        improved = best_fid is None or fid_score < (best_fid - args.min_delta)
        if improved:
            best_fid = fid_score
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": last_epoch,
            "fid_score": fid_score,
            "best_fid": best_fid,
            "improved": improved,
            "no_improve": no_improve,
            "eval_time_sec": round(eval_time, 2),
        }
        write_csv_row(results_csv, list(row.keys()), row)

        with open(results_jsonl, "a") as f:
            f.write(json.dumps({"epoch": last_epoch, **fid_data}) + "\n")

        log_line = (
            f"Eval @ epoch {last_epoch}: FID={fid_score:.4f}, "
            f"best={best_fid:.4f}, no_improve={no_improve}/{args.patience}, "
            f"improved={improved}, eval_time={eval_time:.1f}s"
        )
        print(log_line, flush=True)
        with open(results_log, "a") as f:
            f.write(log_line + "\n")

        # Persist early-stopping state for crash recovery
        with open(state_path, "w") as f:
            json.dump({"best_fid": best_fid, "no_improve": no_improve, "done": False}, f)

        if no_improve >= args.patience:
            print("Early stopping: no improvement.", flush=True)
            with open(state_path, "w") as f:
                json.dump({"best_fid": best_fid, "no_improve": no_improve, "done": True}, f)
            break

    else:
        # Reached max_epochs without early stopping
        with open(state_path, "w") as f:
            json.dump({"best_fid": best_fid, "no_improve": no_improve, "done": True}, f)


if __name__ == "__main__":
    main()
