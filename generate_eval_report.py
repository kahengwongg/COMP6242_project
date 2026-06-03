"""
Generate evaluation artifacts for all 12 experiments:
1) CSV summary
2) comparison tables
3) plots
4) markdown report

Usage:
    python generate_eval_report.py --exp_root experiments --out_dir reports
"""

import os
import csv
import json
import argparse
import glob
from statistics import mean

import matplotlib.pyplot as plt


MODELS = ["dcgan", "wgan_gp", "attention_gan", "combined",
          "attention_gan_g_bounded", "attention_gan_gd_bounded"]
CONDITIONS = ["full_data", "low_data", "noisy"]


def to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_experiment_row(exp_root, model, condition, seed, dataset_tag):
    exp_name = f"{model}_{dataset_tag}_{condition}_seed{seed}"
    exp_dir = os.path.join(exp_root, exp_name)

    fid_path = os.path.join(exp_dir, "fid_results.json")
    loss_path = os.path.join(exp_dir, "loss_stats.json")

    fid = None
    method = None
    kid_mean = None
    kid_std = None
    is_mean = None
    is_std = None
    precision = None
    recall = None
    ms_ssim_mean = None
    ms_ssim_diversity = None
    lpips_div_mean = None
    lpips_div_std = None
    mifid_proxy = None
    nn_lpips_mean = None
    memorization_risk = None
    g_std = None
    d_std = None
    hours = None
    final_g = None
    final_d = None

    if os.path.exists(fid_path):
        with open(fid_path, "r", encoding="utf-8") as f:
            fid_data = json.load(f)
        fid = to_float(fid_data.get("fid_score"))
        method = fid_data.get("fid_method")
        kid_mean = to_float(fid_data.get("kid_mean"))
        kid_std = to_float(fid_data.get("kid_std"))
        is_mean = to_float(fid_data.get("inception_score_mean"))
        is_std = to_float(fid_data.get("inception_score_std"))
        precision = to_float(fid_data.get("precision"))
        recall = to_float(fid_data.get("recall"))
        ms_ssim_mean = to_float(fid_data.get("ms_ssim_mean"))
        ms_ssim_diversity = to_float(fid_data.get("ms_ssim_diversity"))
        lpips_div_mean = to_float(fid_data.get("lpips_diversity_mean"))
        lpips_div_std = to_float(fid_data.get("lpips_diversity_std"))
        mifid_proxy = to_float(fid_data.get("mifid_proxy"))
        nn_lpips_mean = to_float(fid_data.get("nn_lpips_mean"))
        memorization_risk = to_float(fid_data.get("memorization_risk"))

    if os.path.exists(loss_path):
        with open(loss_path, "r", encoding="utf-8") as f:
            loss_data = json.load(f)
        g_std = to_float(loss_data.get("generator", {}).get("std"))
        d_std = to_float(loss_data.get("discriminator", {}).get("std"))
        hours = to_float(loss_data.get("total_time_hours"))
        final_g = to_float(loss_data.get("final_g_loss"))
        final_d = to_float(loss_data.get("final_d_loss"))

    return {
        "experiment": exp_name,
        "model": model,
        "condition": condition,
        "seed": seed,
        "fid": fid,
        "fid_method": method,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "precision": precision,
        "recall": recall,
        "ms_ssim_mean": ms_ssim_mean,
        "ms_ssim_diversity": ms_ssim_diversity,
        "lpips_diversity_mean": lpips_div_mean,
        "lpips_diversity_std": lpips_div_std,
        "mifid_proxy": mifid_proxy,
        "nn_lpips_mean": nn_lpips_mean,
        "memorization_risk": memorization_risk,
        "g_std": g_std,
        "d_std": d_std,
        "total_time_hours": hours,
        "final_g_loss": final_g,
        "final_d_loss": final_d,
        "exp_dir": exp_dir,
    }


def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = [
        "experiment",
        "model",
        "condition",
        "seed",
        "fid",
        "fid_method",
        "kid_mean",
        "kid_std",
        "inception_score_mean",
        "inception_score_std",
        "precision",
        "recall",
        "ms_ssim_mean",
        "ms_ssim_diversity",
        "lpips_diversity_mean",
        "lpips_diversity_std",
        "mifid_proxy",
        "nn_lpips_mean",
        "memorization_risk",
        "g_std",
        "d_std",
        "total_time_hours",
        "final_g_loss",
        "final_d_loss",
        "exp_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def read_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["fid"] = to_float(r.get("fid"))
            r["g_std"] = to_float(r.get("g_std"))
            r["d_std"] = to_float(r.get("d_std"))
            r["kid_mean"] = to_float(r.get("kid_mean"))
            r["kid_std"] = to_float(r.get("kid_std"))
            r["inception_score_mean"] = to_float(r.get("inception_score_mean"))
            r["inception_score_std"] = to_float(r.get("inception_score_std"))
            r["precision"] = to_float(r.get("precision"))
            r["recall"] = to_float(r.get("recall"))
            r["ms_ssim_mean"] = to_float(r.get("ms_ssim_mean"))
            r["ms_ssim_diversity"] = to_float(r.get("ms_ssim_diversity"))
            r["lpips_diversity_mean"] = to_float(r.get("lpips_diversity_mean"))
            r["lpips_diversity_std"] = to_float(r.get("lpips_diversity_std"))
            r["mifid_proxy"] = to_float(r.get("mifid_proxy"))
            r["nn_lpips_mean"] = to_float(r.get("nn_lpips_mean"))
            r["memorization_risk"] = to_float(r.get("memorization_risk"))
            r["total_time_hours"] = to_float(r.get("total_time_hours"))
            r["final_g_loss"] = to_float(r.get("final_g_loss"))
            r["final_d_loss"] = to_float(r.get("final_d_loss"))
            rows.append(r)
    return rows


def fmt(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def generate_rank_table(rows):
    valid = [r for r in rows if r["fid"] is not None]
    valid.sort(key=lambda x: x["fid"])
    table = []
    for i, r in enumerate(valid, start=1):
        table.append({
            "rank": i,
            "experiment": r["experiment"],
            "model": r["model"],
            "condition": r["condition"],
            "fid": r["fid"],
            "kid_mean": r.get("kid_mean"),
            "is_mean": r.get("inception_score_mean"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "ms_ssim_diversity": r.get("ms_ssim_diversity"),
            "lpips_diversity_mean": r.get("lpips_diversity_mean"),
            "mifid_proxy": r.get("mifid_proxy"),
            "memorization_risk": r.get("memorization_risk"),
            "g_std": r["g_std"],
            "d_std": r["d_std"],
            "hours": r["total_time_hours"],
        })
    return table


def write_rank_csv(rank_rows, out_path):
    fields = ["rank", "experiment", "model", "condition", "fid", "g_std", "d_std", "hours"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rank_rows:
            writer.writerow(r)


def find_sample_image(exp_dir):
    """Return a representative generated image path for one experiment, or None."""
    # Prefer FID samples (always generated during evaluate.py)
    fid_candidates = sorted(glob.glob(os.path.join(exp_dir, "fid_samples", "*.png")))
    if fid_candidates:
        return fid_candidates[0]

    # Fallback to training snapshots if available
    sample_candidates = sorted(glob.glob(os.path.join(exp_dir, "samples", "*.png")))
    if sample_candidates:
        return sample_candidates[-1]

    return None


def build_sample_showcase(rows, out_dir):
    """Build a 4x3 grid figure with one representative generated image per experiment."""
    grid_path = os.path.join(out_dir, "generated_samples_grid.png")

    fig, axes = plt.subplots(len(MODELS), len(CONDITIONS), figsize=(12, 12))
    fig.suptitle("Generated Samples Showcase (one sample per experiment)", fontsize=14)

    # Ensure consistent placement by model/condition
    for i, model in enumerate(MODELS):
        for j, condition in enumerate(CONDITIONS):
            ax = axes[i, j]
            match = [r for r in rows if r["model"] == model and r["condition"] == condition]
            row = match[0] if match else None

            ax.set_xticks([])
            ax.set_yticks([])

            if row is None:
                ax.set_title(f"{model} | {condition}", fontsize=9)
                ax.text(0.5, 0.5, "Missing row", ha="center", va="center", fontsize=9)
                continue

            sample_path = find_sample_image(row["exp_dir"])
            row["sample_image"] = sample_path

            ax.set_title(f"{model} | {condition}\nFID={fmt(row['fid'], 2)}", fontsize=8)
            if sample_path and os.path.exists(sample_path):
                img = plt.imread(sample_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "No sample image", ha="center", va="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(grid_path, dpi=180)
    plt.close()

    return grid_path


def plot_fid_bar(rank_rows, out_path):
    names = [r["experiment"] for r in rank_rows]
    vals = [r["fid"] for r in rank_rows]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=55, ha="right", fontsize=8)
    plt.ylabel("FID (lower is better)")
    plt.title("FID Ranking Across 12 Experiments")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_fid_heatmap(rows, out_path):
    # matrix shape: model x condition
    matrix = []
    for m in MODELS:
        row = []
        for c in CONDITIONS:
            match = [r for r in rows if r["model"] == m and r["condition"] == c]
            row.append(match[0]["fid"] if match else None)
        matrix.append(row)

    # Replace None with 0 for plotting; annotate as N/A
    numeric = [[v if v is not None else 0.0 for v in row] for row in matrix]

    plt.figure(figsize=(7, 4))
    im = plt.imshow(numeric, aspect="auto")
    plt.colorbar(im, label="FID")
    plt.xticks(range(len(CONDITIONS)), CONDITIONS)
    plt.yticks(range(len(MODELS)), MODELS)
    plt.title("FID Heatmap (Model x Condition)")

    for i in range(len(MODELS)):
        for j in range(len(CONDITIONS)):
            v = matrix[i][j]
            txt = "N/A" if v is None else f"{v:.1f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def model_condition_summary(rows):
    by_model = {}
    by_condition = {}

    for m in MODELS:
        vals = [r["fid"] for r in rows if r["model"] == m and r["fid"] is not None]
        by_model[m] = mean(vals) if vals else None

    for c in CONDITIONS:
        vals = [r["fid"] for r in rows if r["condition"] == c and r["fid"] is not None]
        by_condition[c] = mean(vals) if vals else None

    return by_model, by_condition


def best_row(rows, key, higher_is_better=False):
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda x: x[key]) if higher_is_better else min(valid, key=lambda x: x[key])


def mean_by_model(rows, key):
    out = {}
    for m in MODELS:
        vals = [r.get(key) for r in rows if r.get("model") == m and r.get(key) is not None]
        out[m] = mean(vals) if vals else None
    return out


def to_markdown_table(rank_rows):
    lines = []
    lines.append("| Rank | Experiment | FID | KID | IS | Recall | LPIPS div | 1-MS-SSIM | MiFID proxy | Mem risk | G_std | D_std | Hours |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rank_rows:
        lines.append(
            f"| {r['rank']} | {r['experiment']} | {fmt(r['fid'], 2)} | {fmt(r.get('kid_mean'), 6)} | {fmt(r.get('is_mean'), 3)} | {fmt(r.get('recall'), 4)} | {fmt(r.get('lpips_diversity_mean'), 4)} | {fmt(r.get('ms_ssim_diversity'), 4)} | {fmt(r.get('mifid_proxy'), 2)} | {fmt(r.get('memorization_risk'), 4)} | {fmt(r['g_std'], 4)} | {fmt(r['d_std'], 4)} | {fmt(r['hours'], 2)} |"
        )
    return "\n".join(lines)


def write_markdown_report(rows, rank_rows, out_md, csv_path, plot_bar_path, plot_heatmap_path, sample_grid_path):
    valid = [r for r in rows if r["fid"] is not None]
    best = min(valid, key=lambda x: x["fid"]) if valid else None
    worst = max(valid, key=lambda x: x["fid"]) if valid else None

    stable_rows = [r for r in rows if r["g_std"] is not None]
    most_stable = min(stable_rows, key=lambda x: x["g_std"]) if stable_rows else None
    best_kid_rows = [r for r in rows if r.get("kid_mean") is not None]
    best_kid = min(best_kid_rows, key=lambda x: x["kid_mean"]) if best_kid_rows else None
    best_precision_rows = [r for r in rows if r.get("precision") is not None]
    best_precision = max(best_precision_rows, key=lambda x: x["precision"]) if best_precision_rows else None
    best_recall_rows = [r for r in rows if r.get("recall") is not None]
    best_recall = max(best_recall_rows, key=lambda x: x["recall"]) if best_recall_rows else None
    best_lpips_div_rows = [r for r in rows if r.get("lpips_diversity_mean") is not None]
    best_lpips_div = max(best_lpips_div_rows, key=lambda x: x["lpips_diversity_mean"]) if best_lpips_div_rows else None
    best_msssim_div_rows = [r for r in rows if r.get("ms_ssim_diversity") is not None]
    best_msssim_div = max(best_msssim_div_rows, key=lambda x: x["ms_ssim_diversity"]) if best_msssim_div_rows else None
    best_mifid_rows = [r for r in rows if r.get("mifid_proxy") is not None]
    best_mifid = min(best_mifid_rows, key=lambda x: x["mifid_proxy"]) if best_mifid_rows else None

    by_model, by_condition = model_condition_summary(rows)
    by_model_kid = mean_by_model(rows, "kid_mean")
    by_model_recall = mean_by_model(rows, "recall")
    by_model_lpips_div = mean_by_model(rows, "lpips_diversity_mean")
    by_model_mifid = mean_by_model(rows, "mifid_proxy")

    b_fid = best_row(rows, "fid", higher_is_better=False)
    b_kid = best_row(rows, "kid_mean", higher_is_better=False)
    b_recall = best_row(rows, "recall", higher_is_better=True)
    b_lpips = best_row(rows, "lpips_diversity_mean", higher_is_better=True)
    b_msssim_div = best_row(rows, "ms_ssim_diversity", higher_is_better=True)
    b_mifid = best_row(rows, "mifid_proxy", higher_is_better=False)

    md = []
    md.append("# 12-Experiment Evaluation Report")
    md.append("")
    md.append("## Data Files")
    md.append("")
    md.append(f"- Raw merged CSV: {os.path.basename(csv_path)}")
    md.append("- Ranking table is included directly in this report")
    md.append("")
    md.append("## Comparison Table (from CSV)")
    md.append("")
    md.append(to_markdown_table(rank_rows))
    md.append("")
    md.append("## Plots")
    md.append("")
    md.append(f"![FID Ranking]({os.path.basename(plot_bar_path)})")
    md.append("")
    md.append(f"![FID Heatmap]({os.path.basename(plot_heatmap_path)})")
    md.append("")
    md.append("## Generated Image Results")
    md.append("")
    md.append("### Overview Grid")
    md.append("")
    md.append(f"![Generated Samples Grid]({os.path.basename(sample_grid_path)})")
    md.append("")
    md.append("### Per-Experiment Sample")
    md.append("")
    for model in MODELS:
        for condition in CONDITIONS:
            match = [r for r in rows if r["model"] == model and r["condition"] == condition]
            if not match:
                continue
            r = match[0]
            md.append(f"- {r['experiment']} (FID: {fmt(r['fid'], 2)})")
            sample_path = r.get("sample_image")
            if sample_path and os.path.exists(sample_path):
                rel = os.path.relpath(sample_path, os.path.dirname(out_md))
                md.append(f"  ![]({rel})")
            else:
                md.append("  (No sample image found)")
    md.append("")
    md.append("## Comparison")
    md.append("")
    md.append("### Average FID by Model")
    md.append("")
    for k, v in by_model.items():
        md.append(f"- {k}: {fmt(v, 2)}")
    md.append("")
    md.append("### Average FID by Condition")
    md.append("")
    for k, v in by_condition.items():
        md.append(f"- {k}: {fmt(v, 2)}")
    md.append("")
    md.append("## Metric-by-Metric Interpretation")
    md.append("")
    md.append("### 1) Fidelity / Quality (FID, KID)")
    md.append("")
    md.append("- Interpretation: lower is better; indicates generated distribution is closer to real data.")
    if b_fid:
        md.append(f"- Best FID experiment: {b_fid['experiment']} ({fmt(b_fid['fid'], 2)}).")
    if b_kid:
        md.append(f"- Best KID experiment: {b_kid['experiment']} ({fmt(b_kid['kid_mean'], 6)}).")
    md.append("- Model-level view (mean KID):")
    for m in MODELS:
        md.append(f"  - {m}: {fmt(by_model_kid.get(m), 6)}")
    md.append("")
    md.append("### 2) Coverage / Diversity (Recall, LPIPS, 1-MS-SSIM)")
    md.append("")
    md.append("- Interpretation: higher Recall and higher diversity statistics usually indicate broader mode coverage.")
    if b_recall:
        md.append(f"- Best Recall experiment: {b_recall['experiment']} ({fmt(b_recall['recall'], 4)}).")
    if b_lpips:
        md.append(f"- Best LPIPS diversity experiment: {b_lpips['experiment']} ({fmt(b_lpips['lpips_diversity_mean'], 4)}).")
    if b_msssim_div:
        md.append(f"- Best 1-MS-SSIM diversity experiment: {b_msssim_div['experiment']} ({fmt(b_msssim_div['ms_ssim_diversity'], 4)}).")
    md.append("- Model-level view:")
    for m in MODELS:
        md.append(
            f"  - {m}: Recall={fmt(by_model_recall.get(m), 4)}, LPIPS-div={fmt(by_model_lpips_div.get(m), 4)}"
        )
    md.append("")
    md.append("### 3) Anti-Memorization (MiFID proxy)")
    md.append("")
    md.append("- Interpretation: lower MiFID proxy is better; it combines quality with nearest-neighbor overfitting risk.")
    md.append("- Caveat: this is a practical proxy, not a strict theorem-level memorization proof.")
    if b_mifid:
        md.append(f"- Best MiFID proxy experiment: {b_mifid['experiment']} ({fmt(b_mifid['mifid_proxy'], 2)}).")
    md.append("- Model-level view (mean MiFID proxy):")
    for m in MODELS:
        md.append(f"  - {m}: {fmt(by_model_mifid.get(m), 2)}")
    md.append("")
    md.append("### 4) Stability (G_std / D_std)")
    md.append("")
    md.append("- Interpretation: lower std suggests more stable optimization dynamics.")
    if most_stable:
        md.append(f"- Most stable experiment by G_std: {most_stable['experiment']} ({fmt(most_stable['g_std'], 4)}).")
    md.append("")
    md.append("### 5) Practical Model Takeaways")
    md.append("")
    md.append("- Fidelity-oriented selection is supported by lower FID/KID; in this benchmark, the strongest settings are concentrated in WGAN-GP/Combined under full_data.")
    md.append("- Diversity-oriented assessment should jointly consider Recall, LPIPS-div, and 1-MS-SSIM-div, rather than relying on FID alone.")
    md.append("- Final model choice is best treated as a Pareto trade-off across fidelity, diversity, and memorization risk.")
    md.append("")
    md.append("## Conclusions")
    md.append("")
    if best:
        md.append(f"- Best FID: {best['experiment']} ({fmt(best['fid'], 2)}).")
    if worst:
        md.append(f"- Worst FID: {worst['experiment']} ({fmt(worst['fid'], 2)}).")
    if most_stable:
        md.append(f"- Most stable (lowest G_std): {most_stable['experiment']} ({fmt(most_stable['g_std'], 4)}).")
    if best_kid:
        md.append(f"- Best KID (lower is better): {best_kid['experiment']} ({fmt(best_kid['kid_mean'], 6)}).")
    if best_precision:
        md.append(f"- Best Precision (higher is better): {best_precision['experiment']} ({fmt(best_precision['precision'], 4)}).")
    if best_recall:
        md.append(f"- Best Recall (higher is better): {best_recall['experiment']} ({fmt(best_recall['recall'], 4)}).")
    if best_lpips_div:
        md.append(f"- Best LPIPS diversity (higher is better): {best_lpips_div['experiment']} ({fmt(best_lpips_div['lpips_diversity_mean'], 4)}).")
    if best_msssim_div:
        md.append(f"- Best 1-MS-SSIM diversity (higher is better): {best_msssim_div['experiment']} ({fmt(best_msssim_div['ms_ssim_diversity'], 4)}).")
    if best_mifid:
        md.append(f"- Best MiFID proxy (lower is better): {best_mifid['experiment']} ({fmt(best_mifid['mifid_proxy'], 2)}).")
    md.append("- Lower FID indicates better generation quality and closer real-data distribution.")
    md.append("- MiFID proxy is a practical anti-memorization proxy: it penalizes low nearest-neighbor LPIPS distance to real images.")
    md.append("- Use the ranking table for final model selection and combine it with stability metrics (G_std, D_std).")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="experiments")
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--dataset_tag", default="celeba")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: collect and write raw csv
    rows = []
    for m in MODELS:
        for c in CONDITIONS:
            rows.append(load_experiment_row(args.exp_root, m, c, args.seed, args.dataset_tag))

    csv_path = os.path.join(args.out_dir, "eval_results.csv")
    write_csv(rows, csv_path)

    # Step 2: read from csv, then produce rank table and plots
    rows_from_csv = read_csv(csv_path)
    rank_rows = generate_rank_table(rows_from_csv)

    fid_bar_path = os.path.join(args.out_dir, "fid_ranking_bar.png")
    fid_heatmap_path = os.path.join(args.out_dir, "fid_heatmap.png")
    sample_grid_path = os.path.join(args.out_dir, "generated_samples_grid.png")
    plot_fid_bar(rank_rows, fid_bar_path)
    plot_fid_heatmap(rows_from_csv, fid_heatmap_path)
    build_sample_showcase(rows_from_csv, args.out_dir)

    # Step 3: markdown report
    md_path = os.path.join(args.out_dir, "eval_report.md")
    write_markdown_report(
        rows_from_csv,
        rank_rows,
        md_path,
        csv_path,
        fid_bar_path,
        fid_heatmap_path,
        sample_grid_path,
    )

    print("Generated artifacts:")
    print(f"- {csv_path}")
    print(f"- {fid_bar_path}")
    print(f"- {fid_heatmap_path}")
    print(f"- {sample_grid_path}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()
