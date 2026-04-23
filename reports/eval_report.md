# 12-Experiment Evaluation Report

## Data Files

- Raw merged CSV: eval_results.csv
- Ranking table is included directly in this report

## Comparison Table (from CSV)

| Rank | Experiment | FID | KID | IS | Recall | LPIPS div | 1-MS-SSIM | MiFID proxy | Mem risk | G_std | D_std | Hours |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | wgan_gp_celeba_full_data_seed42 | 175.20 | 0.208769 | 2.298 | 0.3308 | 0.2842 | 0.6972 | 175.20 | 0.0000 | 4.3217 | 1.8545 | 0.91 |
| 2 | combined_celeba_full_data_seed42 | 177.46 | 0.208616 | 2.298 | 0.2498 | 0.2897 | 0.7396 | 177.46 | 0.0000 | 6.2429 | 1.7848 | 1.23 |
| 3 | dcgan_celeba_full_data_seed42 | 180.95 | 0.217536 | 2.298 | 0.3572 | 0.2794 | 0.6936 | 180.95 | 0.0000 | 2.0187 | 0.4723 | 0.76 |
| 4 | dcgan_celeba_low_data_seed42 | 189.15 | 0.230173 | 2.298 | 0.3958 | 0.2676 | 0.6897 | 189.15 | 0.0000 | 1.4270 | 0.5520 | 0.08 |
| 5 | dcgan_celeba_noisy_seed42 | 207.53 | 0.247987 | 2.298 | 0.1318 | 0.2959 | 0.7480 | 207.53 | 0.0000 | 1.5036 | 0.3601 | 0.76 |
| 6 | attention_gan_celeba_low_data_seed42 | 210.08 | 0.259095 | 2.298 | 0.3150 | 0.2704 | 0.6738 | 210.08 | 0.0000 | 1.5837 | 0.4386 | 0.14 |
| 7 | attention_gan_celeba_noisy_seed42 | 212.17 | 0.251974 | 2.298 | 0.1056 | 0.2936 | 0.7237 | 212.17 | 0.0000 | 1.8422 | 0.3706 | 1.36 |
| 8 | wgan_gp_celeba_noisy_seed42 | 216.12 | 0.259927 | 2.298 | 0.1448 | 0.2974 | 0.7592 | 216.12 | 0.0000 | 1.7532 | 0.3134 | 0.56 |
| 9 | combined_celeba_noisy_seed42 | 219.42 | 0.263394 | 2.298 | 0.1176 | 0.2752 | 0.7529 | 219.42 | 0.0000 | 3.5112 | 1.6513 | 1.23 |
| 10 | wgan_gp_celeba_low_data_seed42 | 222.18 | 0.280367 | 2.298 | 0.1686 | 0.2700 | 0.7386 | 222.18 | 0.0000 | 8.6579 | 4.1293 | 0.09 |
| 11 | combined_celeba_low_data_seed42 | 226.69 | 0.283707 | 2.298 | 0.1980 | 0.2788 | 0.7402 | 226.69 | 0.0000 | N/A | N/A | N/A |
| 12 | attention_gan_celeba_full_data_seed42 | 321.71 | 0.428627 | 2.298 | 0.0002 | 0.2432 | 0.3315 | 321.71 | 0.0000 | 35.1192 | 0.6022 | 1.36 |

## Plots

![FID Ranking](fid_ranking_bar.png)

![FID Heatmap](fid_heatmap.png)

## Generated Image Results

### Overview Grid

![Generated Samples Grid](generated_samples_grid.png)

### Per-Experiment Sample

- dcgan_celeba_full_data_seed42 (FID: 180.95)
  ![](../experiments/dcgan_celeba_full_data_seed42/fid_samples/00000.png)
- dcgan_celeba_low_data_seed42 (FID: 189.15)
  ![](../experiments/dcgan_celeba_low_data_seed42/fid_samples/00000.png)
- dcgan_celeba_noisy_seed42 (FID: 207.53)
  ![](../experiments/dcgan_celeba_noisy_seed42/fid_samples/00000.png)
- wgan_gp_celeba_full_data_seed42 (FID: 175.20)
  ![](../experiments/wgan_gp_celeba_full_data_seed42/fid_samples/00000.png)
- wgan_gp_celeba_low_data_seed42 (FID: 222.18)
  ![](../experiments/wgan_gp_celeba_low_data_seed42/fid_samples/00000.png)
- wgan_gp_celeba_noisy_seed42 (FID: 216.12)
  ![](../experiments/wgan_gp_celeba_noisy_seed42/fid_samples/00000.png)
- attention_gan_celeba_full_data_seed42 (FID: 321.71)
  ![](../experiments/attention_gan_celeba_full_data_seed42/fid_samples/00000.png)
- attention_gan_celeba_low_data_seed42 (FID: 210.08)
  ![](../experiments/attention_gan_celeba_low_data_seed42/fid_samples/00000.png)
- attention_gan_celeba_noisy_seed42 (FID: 212.17)
  ![](../experiments/attention_gan_celeba_noisy_seed42/fid_samples/00000.png)
- combined_celeba_full_data_seed42 (FID: 177.46)
  ![](../experiments/combined_celeba_full_data_seed42/fid_samples/00000.png)
- combined_celeba_low_data_seed42 (FID: 226.69)
  ![](../experiments/combined_celeba_low_data_seed42/fid_samples/00000.png)
- combined_celeba_noisy_seed42 (FID: 219.42)
  ![](../experiments/combined_celeba_noisy_seed42/fid_samples/00000.png)

## Comparison

### Average FID by Model

- dcgan: 192.54
- wgan_gp: 204.50
- attention_gan: 247.99
- combined: 207.86

### Average FID by Condition

- full_data: 213.83
- low_data: 212.02
- noisy: 213.81

## Metric-by-Metric Interpretation

### 1) Fidelity / Quality (FID, KID)

- Interpretation: lower is better; indicates generated distribution is closer to real data.
- Best FID experiment: wgan_gp_celeba_full_data_seed42 (175.20).
- Best KID experiment: combined_celeba_full_data_seed42 (0.208616).
- Model-level view (mean KID):
  - dcgan: 0.231899
  - wgan_gp: 0.249688
  - attention_gan: 0.313232
  - combined: 0.251906

### 2) Coverage / Diversity (Recall, LPIPS, 1-MS-SSIM)

- Interpretation: higher Recall and higher diversity statistics usually indicate broader mode coverage.
- Best Recall experiment: dcgan_celeba_low_data_seed42 (0.3958).
- Best LPIPS diversity experiment: wgan_gp_celeba_noisy_seed42 (0.2974).
- Best 1-MS-SSIM diversity experiment: wgan_gp_celeba_noisy_seed42 (0.7592).
- Model-level view:
  - dcgan: Recall=0.2949, LPIPS-div=0.2810
  - wgan_gp: Recall=0.2147, LPIPS-div=0.2838
  - attention_gan: Recall=0.1403, LPIPS-div=0.2691
  - combined: Recall=0.1885, LPIPS-div=0.2812

### 3) Anti-Memorization (MiFID proxy)

- Interpretation: lower MiFID proxy is better; it combines quality with nearest-neighbor overfitting risk.
- Caveat: this is a practical proxy, not a strict theorem-level memorization proof.
- Best MiFID proxy experiment: wgan_gp_celeba_full_data_seed42 (175.20).
- Model-level view (mean MiFID proxy):
  - dcgan: 192.54
  - wgan_gp: 204.50
  - attention_gan: 247.99
  - combined: 207.86

### 4) Stability (G_std / D_std)

- Interpretation: lower std suggests more stable optimization dynamics.
- Most stable experiment by G_std: dcgan_celeba_low_data_seed42 (1.4270).

### 5) Practical Model Takeaways

- Fidelity-oriented selection is supported by lower FID/KID; in this benchmark, the strongest settings are concentrated in WGAN-GP/Combined under full_data.
- Diversity-oriented assessment should jointly consider Recall, LPIPS-div, and 1-MS-SSIM-div, rather than relying on FID alone.
- Final model choice is best treated as a Pareto trade-off across fidelity, diversity, and memorization risk.

## Conclusions

- Best FID: wgan_gp_celeba_full_data_seed42 (175.20).
- Worst FID: attention_gan_celeba_full_data_seed42 (321.71).
- Most stable (lowest G_std): dcgan_celeba_low_data_seed42 (1.4270).
- Best KID (lower is better): combined_celeba_full_data_seed42 (0.208616).
- Best Precision (higher is better): dcgan_celeba_full_data_seed42 (0.0000).
- Best Recall (higher is better): dcgan_celeba_low_data_seed42 (0.3958).
- Best LPIPS diversity (higher is better): wgan_gp_celeba_noisy_seed42 (0.2974).
- Best 1-MS-SSIM diversity (higher is better): wgan_gp_celeba_noisy_seed42 (0.7592).
- Best MiFID proxy (lower is better): wgan_gp_celeba_full_data_seed42 (175.20).
- Lower FID indicates better generation quality and closer real-data distribution.
- MiFID proxy is a practical anti-memorization proxy: it penalizes low nearest-neighbor LPIPS distance to real images.
- Use the ranking table for final model selection and combine it with stability metrics (G_std, D_std).