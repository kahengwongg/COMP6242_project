"""
Run FID evaluation for all 12 CelebA experiments and print a comparison table.

Usage:
    python compare_experiments.py --data_dir data/celeba --num_samples 5000
"""

import os
import json
import argparse
import subprocess
import sys


MODELS = ['dcgan', 'wgan_gp', 'attention_gan', 'combined']
CONDITIONS = ['full_data', 'low_data', 'noisy']


def run_fid(exp_dir, data_dir, num_samples, device):
    cmd = [
        sys.executable, 'evaluate.py',
        '--exp_dir', exp_dir,
        '--data_dir', data_dir,
        '--num_samples', str(num_samples),
        '--device', device,
    ]
    print(f'\n>>> {" ".join(cmd)}')
    subprocess.run(cmd, check=False)


def load_results(exp_dir):
    fid_path = os.path.join(exp_dir, 'fid_results.json')
    loss_path = os.path.join(exp_dir, 'loss_stats.json')
    fid = fid_val = None
    kid = is_mean = precision = recall = None
    lpips_div = msssim_div = mifid_proxy = mem_risk = None
    g_std = d_std = hours = None

    if os.path.exists(fid_path):
        with open(fid_path) as f:
            data = json.load(f)
        fid_val = data.get('fid_score')
        kid = data.get('kid_mean')
        is_mean = data.get('inception_score_mean')
        precision = data.get('precision')
        recall = data.get('recall')
        lpips_div = data.get('lpips_diversity_mean')
        msssim_div = data.get('ms_ssim_diversity')
        mifid_proxy = data.get('mifid_proxy')
        mem_risk = data.get('memorization_risk')

    if os.path.exists(loss_path):
        with open(loss_path) as f:
            data = json.load(f)
        g_std = data.get('generator', {}).get('std')
        d_std = data.get('discriminator', {}).get('std')
        hours = data.get('total_time_hours')

    return fid_val, kid, is_mean, precision, recall, lpips_div, msssim_div, mifid_proxy, mem_risk, g_std, d_std, hours


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/celeba')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip FID computation, only print table from cached results')
    parser.add_argument('--dataset_tag', default='celeba',
                        help='Dataset tag used in experiment directory names')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    tag = args.dataset_tag
    seed = args.seed

    # ── Phase 1: run evaluate.py for each experiment ───────────────────────
    if not args.skip_eval:
        for model in MODELS:
            for condition in CONDITIONS:
                exp_dir = f'experiments/{model}_{tag}_{condition}_seed{seed}'
                if not os.path.isdir(exp_dir):
                    print(f'[skip] {exp_dir} not found')
                    continue
                # skip if FID already computed
                fid_path = os.path.join(exp_dir, 'fid_results.json')
                if os.path.exists(fid_path):
                    print(f'[cached] {exp_dir}')
                    continue
                run_fid(exp_dir, args.data_dir, args.num_samples, args.device)

    # ── Phase 2: print comparison table ────────────────────────────────────
    print('\n')
    print('=' * 170)
    print(f'{"Experiment":<33} {"FID ↓":>9} {"KID ↓":>10} {"IS ↑":>8} {"Rec ↑":>8} {"LPIPS ↑":>9} {"1-MSSSIM ↑":>11} {"MiFID ↓":>10} {"MemRisk ↓":>10} {"G_std":>8} {"D_std":>8} {"Hours":>8}')
    print('=' * 170)

    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            exp_dir = f'experiments/{model}_{tag}_{condition}_seed{seed}'
            fid_val, kid, is_mean, precision, recall, lpips_div, msssim_div, mifid_proxy, mem_risk, g_std, d_std, hours = load_results(exp_dir)
            rows.append((model, condition, fid_val, kid, is_mean, precision, recall, lpips_div, msssim_div, mifid_proxy, mem_risk, g_std, d_std, hours))

            fid_str  = f'{fid_val:.2f}' if fid_val  is not None else 'N/A'
            kid_str  = f'{kid:.6f}' if kid is not None else 'N/A'
            is_str   = f'{is_mean:.3f}' if is_mean is not None else 'N/A'
            r_str    = f'{recall:.4f}' if recall is not None else 'N/A'
            lp_str   = f'{lpips_div:.4f}' if lpips_div is not None else 'N/A'
            ms_str   = f'{msssim_div:.4f}' if msssim_div is not None else 'N/A'
            mf_str   = f'{mifid_proxy:.2f}' if mifid_proxy is not None else 'N/A'
            mr_str   = f'{mem_risk:.4f}' if mem_risk is not None else 'N/A'
            g_str    = f'{g_std:.4f}'   if g_std    is not None else 'N/A'
            d_str    = f'{d_std:.4f}'   if d_std    is not None else 'N/A'
            h_str    = f'{hours:.2f}'   if hours    is not None else 'N/A'
            name = f'{model}_{condition}'
            print(f'{name:<33} {fid_str:>9} {kid_str:>10} {is_str:>8} {r_str:>8} {lp_str:>9} {ms_str:>11} {mf_str:>10} {mr_str:>10} {g_str:>8} {d_str:>8} {h_str:>8}')

        print('-' * 170)

    # ── Phase 3: summary insights ───────────────────────────────────────────
    valid = [(m, c, f, k, i, p, r, lp, ms, mf, mr, gs, ds, h) for m, c, f, k, i, p, r, lp, ms, mf, mr, gs, ds, h in rows if f is not None]
    if valid:
        best = min(valid, key=lambda x: x[2])
        worst = max(valid, key=lambda x: x[2])
        most_stable = min(valid, key=lambda x: (x[11] or 1e9))
        print(f'\nBest  FID : {best[0]}_{best[1]}  →  {best[2]:.2f}')
        print(f'Worst FID : {worst[0]}_{worst[1]}  →  {worst[2]:.2f}')
        print(f'Most stable (lowest G_std): {most_stable[0]}_{most_stable[1]}  →  G_std={most_stable[11]:.4f}')

    # ── Phase 4: save summary JSON ─────────────────────────────────────────
    summary = []
    for model, condition, fid_val, kid, is_mean, precision, recall, lpips_div, msssim_div, mifid_proxy, mem_risk, g_std, d_std, hours in rows:
        summary.append({
            'model': model, 'condition': condition,
            'fid': fid_val, 'kid': kid, 'is': is_mean,
            'precision': precision, 'recall': recall,
            'lpips_diversity': lpips_div,
            'ms_ssim_diversity': msssim_div,
            'mifid_proxy': mifid_proxy,
            'memorization_risk': mem_risk,
            'g_std': g_std, 'd_std': d_std, 'hours': hours,
        })
    with open('experiments/comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nSummary saved to experiments/comparison_summary.json')


if __name__ == '__main__':
    main()
