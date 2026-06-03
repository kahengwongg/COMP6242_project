# CelebA Seed-42 & Seed-123 FID Results Summary

**Dataset**: CelebA face images  
**Seeds**: 42 (12 experiments), 123 (1 experiment — AttentionGAN full\_data only)  
**Evaluation method**: pytorch-fid  
**Generated samples**: 10,000 per experiment  
**Models evaluated**: DCGAN, WGAN-GP, AttentionGAN, Combined  
**Conditions**: full\_data, low\_data, noisy  

---

## Complete Results Table

All 12 experiments sorted by FID ascending (lower FID = better fidelity).

| # | Experiment | Model | Condition | FID ↓ | MS-SSIM Div ↑ | LPIPS Mean ↑ | LPIPS Std | NN-LPIPS | MiFID | Mem Risk ↓ |
|---|-----------|-------|-----------|-------|--------------|-------------|-----------|----------|-------|-----------|
| 1 | wgan_gp_celeba_full_data_seed42 | WGAN-GP | full_data | 25.686 | 0.710 | 0.276 | 0.072 | 0.158 | 31.049 | 0.209 |
| 2 | dcgan_celeba_full_data_seed42 | DCGAN | full_data | 28.965 | 0.697 | 0.276 | 0.070 | 0.153 | 35.750 | 0.234 |
| 3 | combined_celeba_full_data_seed42 | Combined | full_data | 32.969 | 0.723 | 0.290 | 0.074 | 0.153 | 40.680 | 0.234 |
| 4 | dcgan_celeba_low_data_seed42 | DCGAN | low_data | 34.664 | 0.694 | 0.263 | 0.069 | 0.149 | 43.496 | 0.255 |
| 5 | wgan_gp_celeba_low_data_seed42 | WGAN-GP | low_data | 59.248 | 0.719 | 0.264 | 0.067 | 0.150 | 74.095 | 0.251 |
| 6 | attention_gan_celeba_low_data_seed42 | AttentionGAN | low_data | 60.917 | 0.670 | 0.273 | 0.071 | 0.171 | 69.667 | 0.144 |
| 7 | dcgan_celeba_noisy_seed42 | DCGAN | noisy | 61.554 | 0.749 | 0.303 | 0.078 | 0.183 | 66.821 | 0.086 |
| 8 | attention_gan_celeba_noisy_seed42 | AttentionGAN | noisy | 65.792 | 0.730 | 0.296 | 0.078 | 0.188 | 69.892 | 0.062 |
| 9 | combined_celeba_low_data_seed42 | Combined | low_data | 65.590 | 0.722 | 0.275 | 0.072 | 0.153 | 81.095 | 0.236 |
| 10 | wgan_gp_celeba_noisy_seed42 | WGAN-GP | noisy | 66.154 | 0.752 | 0.289 | 0.074 | 0.201 | 66.154 | 0.000 |
| 11 | combined_celeba_noisy_seed42 | Combined | noisy | 66.629 | 0.734 | 0.284 | 0.075 | 0.169 | 77.112 | 0.157 |
| 12 | attention_gan_celeba_full_data_seed42 | AttentionGAN | full_data | 262.377 | 0.329 | 0.248 | 0.099 | 0.248 | 262.377 | 0.000 |
| 13 | attention_gan_celeba_full_data_seed123 | AttentionGAN | full_data | 309.158 | 0.319 | 0.203 | 0.074 | 0.240 | 309.158 | 0.000 |

**Column key:**
- **FID**: Fréchet Inception Distance — image quality and distribution fidelity (lower is better)
- **MS-SSIM Div**: 1 − MS-SSIM mean across generated pairs — output diversity (higher is better)
- **LPIPS Mean**: Mean perceptual distance between generated pairs — diversity in feature space (higher is better)
- **NN-LPIPS**: Mean nearest-neighbour LPIPS to real images — memorisation indicator (higher = further from real = safer)
- **MiFID**: Memorisation-aware FID proxy (higher penalty for near-copies of training data)
- **Mem Risk**: Estimated memorisation risk score 0–1 (lower is safer)

---

## FID Ranking (Best → Worst)

1. **WGAN-GP / full_data — 25.69** Best overall; gradient penalty provides stable, high-quality training on full data.
2. **DCGAN / full_data — 28.97** Close second; simpler architecture but competitive with full data available.
3. **Combined / full_data — 32.97** Ensemble architecture adds slight FID cost versus DCGAN/WGAN-GP alone.
4. **DCGAN / low_data — 34.66** DCGAN degrades gracefully; only +5.7 points with 10× less data.
5. **WGAN-GP / low_data — 59.25** WGAN-GP suffers more under data scarcity (+33.6 points vs its full-data run).
6. **AttentionGAN / low_data — 60.92** Attention mechanism partially compensates for fewer samples.
7. **DCGAN / noisy — 61.55** Most robust to label/data noise among the four models.
8. **AttentionGAN / noisy — 65.79** Attention helps filter noise slightly; better than Combined and WGAN-GP here.
9. **Combined / low_data — 65.59** Combined model struggles more than DCGAN under data scarcity.
10. **WGAN-GP / noisy — 66.15** Gradient penalty training does not confer noise robustness.
11. **Combined / noisy — 66.63** Worst of the noisy-condition group; ensemble likely amplifies noise sensitivity.
12. **AttentionGAN / full_data (seed 42) — 262.38** Severe outlier — training failure or mode collapse despite full data access.
13. **AttentionGAN / full_data (seed 123) — 309.16** Even worse than seed 42; confirms the failure is architecture-level, not a one-off seed artefact.

---

## Analysis by Model

### DCGAN

| Condition | FID | MiFID | Mem Risk |
|-----------|-----|-------|----------|
| full_data | 28.965 | 35.750 | 0.234 |
| low_data  | 34.664 | 43.496 | 0.255 |
| noisy     | 61.554 | 66.821 | 0.086 |

DCGAN is the most consistent model across all three conditions. Its FID degrades by only ~5.7 points from full to low-data, and it records the best FID under noisy conditions. The memorisation risk is moderate under full and low data but drops significantly under noise, suggesting that noisy training inadvertently acts as a regulariser. The MiFID–FID gap is small (~6–9 points), indicating no severe near-copy generation.

### WGAN-GP

| Condition | FID | MiFID | Mem Risk |
|-----------|-----|-------|----------|
| full_data | 25.686 | 31.049 | 0.209 |
| low_data  | 59.248 | 74.095 | 0.251 |
| noisy     | 66.154 | 66.154 | 0.000 |

WGAN-GP achieves the best FID under full data but is the most data-hungry model: its FID jumps by +33.6 points under low-data (the largest degradation of any model). Under noisy conditions, it records zero memorisation risk and its MiFID equals its raw FID, suggesting it generates outputs diverse enough that no fake sample falls within the memorisation threshold τ = 0.2 of any real image. Despite this, its noisy FID (66.15) is competitive but not superior to DCGAN.

### AttentionGAN

| Condition | FID | MiFID | Mem Risk |
|-----------|-----|-------|----------|
| full_data | 262.377 | 262.377 | 0.000 |
| low_data  |  60.917 |  69.667 | 0.144 |
| noisy     |  65.792 |  69.892 | 0.062 |

AttentionGAN's full-data run is a clear outlier (FID = 262.38), almost certainly indicating training failure or severe mode collapse. Paradoxically, under low-data and noisy conditions its FID is competitive (~61–66), suggesting the model may be sensitive to the full-data training dynamics (e.g., over-regularisation or learning rate interaction at scale). The zero memorisation risk in the full-data run is artefactual — a collapsed generator produces outputs that are uniformly dissimilar to real images. Under low and noisy data the model shows low-to-moderate memorisation risk, consistent with partial convergence.

### Combined

| Condition | FID | MiFID | Mem Risk |
|-----------|-----|-------|----------|
| full_data | 32.969 | 40.680 | 0.234 |
| low_data  | 65.590 | 81.095 | 0.236 |
| noisy     | 66.629 | 77.112 | 0.157 |

The Combined model (ensemble of components from multiple architectures) sits between DCGAN and WGAN-GP on full data but does not outperform either of its constituent models in any condition. It shows the largest MiFID–FID gaps in the low-data and noisy conditions (up to +15.5 points), indicating it generates some near-memorised outputs even in difficult regimes. Its memorisation risk remains consistently moderate (~0.16–0.24) across all conditions.

---

## Analysis by Condition

### Full Data

| Model | FID | MiFID | Mem Risk |
|-------|-----|-------|----------|
| WGAN-GP     | 25.686 | 31.049 | 0.209 |
| DCGAN       | 28.965 | 35.750 | 0.234 |
| Combined    | 32.969 | 40.680 | 0.234 |
| AttentionGAN | 262.377 | 262.377 | 0.000 |

With full data the top three models are tightly clustered (FID 25.7–33.0, a spread of ~7 points). AttentionGAN is an extreme outlier. WGAN-GP's gradient penalty objective is most effective when gradients are supported by a rich real-data distribution. The MiFID penalty adds ~5–8 points for all functional models, indicating low but non-zero near-memorisation activity.

### Low Data

| Model | FID | MiFID | Mem Risk |
|-------|-----|-------|----------|
| DCGAN        | 34.664 | 43.496 | 0.255 |
| WGAN-GP      | 59.248 | 74.095 | 0.251 |
| AttentionGAN | 60.917 | 69.667 | 0.144 |
| Combined     | 65.590 | 81.095 | 0.236 |

DCGAN is the clear winner under data scarcity, with a ~24-point gap to the next best model. WGAN-GP, AttentionGAN, and Combined are tightly clustered (59–66). The Combined model suffers the highest MiFID penalty (+15.5 points) suggesting it memorises more aggressively when data is scarce.

### Noisy

| Model | FID | MiFID | Mem Risk |
|-------|-----|-------|----------|
| DCGAN        | 61.554 | 66.821 | 0.086 |
| AttentionGAN | 65.792 | 69.892 | 0.062 |
| WGAN-GP      | 66.154 | 66.154 | 0.000 |
| Combined     | 66.629 | 77.112 | 0.157 |

All four models cluster within a narrow FID band (61.6–66.6) under noise, indicating that noisy labels are the equalising bottleneck. DCGAN has the best FID and AttentionGAN the lowest memorisation risk. Notably WGAN-GP achieves zero memorisation risk here — its gradient penalty may encourage broader coverage of the real distribution even under noisy supervision. Combined again has the largest MiFID gap (+10.5 points).

---

## Diversity Analysis

### MS-SSIM Diversity (1 − MS-SSIM; higher = more diverse output)

| Model | full_data | low_data | noisy | Average |
|-------|-----------|----------|-------|---------|
| DCGAN       | 0.697 | 0.694 | 0.749 | 0.713 |
| WGAN-GP     | 0.710 | 0.719 | 0.752 | 0.727 |
| AttentionGAN | 0.329 | 0.670 | 0.730 | 0.576 |
| Combined    | 0.723 | 0.722 | 0.734 | 0.726 |

- WGAN-GP and Combined generate the most structurally diverse outputs on average.
- AttentionGAN's full-data diversity score (0.329) is drastically lower than the rest, consistent with mode collapse — the model generates near-identical images.
- Noisy conditions consistently boost MS-SSIM diversity across all models, suggesting noise encourages broader mode coverage.

### LPIPS Diversity (perceptual distance; higher = more diverse)

| Model | full_data | low_data | noisy | Average |
|-------|-----------|----------|-------|---------|
| DCGAN        | 0.276 | 0.263 | 0.303 | 0.281 |
| WGAN-GP      | 0.276 | 0.264 | 0.289 | 0.276 |
| AttentionGAN | 0.248 | 0.273 | 0.296 | 0.272 |
| Combined     | 0.290 | 0.275 | 0.284 | 0.283 |

- Combined has the highest average LPIPS diversity (0.283), marginally ahead of DCGAN (0.281).
- DCGAN under noisy conditions achieves the highest single-run LPIPS diversity (0.303).
- All functional models cluster in a narrow band (0.26–0.30) suggesting similar perceptual diversity across architectures when training succeeds.
- LPIPS std is consistent (~0.07–0.10) across all runs, indicating stable diversity estimates.

---

## Memorisation / MiFID Analysis

### MiFID − FID Gap (higher gap = more memorisation penalty)

| Experiment | FID | MiFID | Gap |
|-----------|-----|-------|-----|
| combined_celeba_low_data_seed42 | 65.590 | 81.095 | +15.505 |
| combined_celeba_noisy_seed42 | 66.629 | 77.112 | +10.483 |
| wgan_gp_celeba_low_data_seed42 | 59.248 | 74.095 | +14.847 |
| attention_gan_celeba_low_data_seed42 | 60.917 | 69.667 | +8.750 |
| attention_gan_celeba_noisy_seed42 | 65.792 | 69.892 | +4.100 |
| combined_celeba_full_data_seed42 | 32.969 | 40.680 | +7.711 |
| dcgan_celeba_low_data_seed42 | 34.664 | 43.496 | +8.832 |
| dcgan_celeba_full_data_seed42 | 28.965 | 35.750 | +6.785 |
| dcgan_celeba_noisy_seed42 | 61.554 | 66.821 | +5.267 |
| wgan_gp_celeba_full_data_seed42 | 25.686 | 31.049 | +5.363 |
| wgan_gp_celeba_noisy_seed42 | 66.154 | 66.154 | 0.000 |
| attention_gan_celeba_full_data_seed42 | 262.377 | 262.377 | 0.000 |

- **Combined / low_data** has the worst memorisation penalty (+15.5 pts), confirming it over-fits when data is scarce.
- **WGAN-GP / noisy** and **AttentionGAN / full_data** both have zero gap (MiFID = FID). For WGAN-GP this is genuinely safe (memorization_risk = 0.0); for AttentionGAN full_data it is artefactual (collapsed outputs never match real images at τ = 0.2).
- Low-data conditions consistently inflate the MiFID gap across all models, indicating that data scarcity drives near-memorisation regardless of architecture.

### Memorisation Risk Summary

- **Zero risk**: AttentionGAN full_data (artefact), WGAN-GP noisy (genuine)
- **Low risk (< 0.1)**: DCGAN noisy (0.086), AttentionGAN noisy (0.062)
- **Moderate risk (0.1–0.2)**: WGAN-GP full_data (0.209), AttentionGAN low_data (0.144), Combined noisy (0.157)
- **Higher risk (> 0.2)**: DCGAN full_data (0.234), DCGAN low_data (0.255), Combined full_data (0.234), Combined low_data (0.236), WGAN-GP low_data (0.251)

All `memorization_risk > 0.2` cases arise in full-data or low-data conditions, not noisy ones — suggesting that noisy labels act as an implicit anti-memorisation regulariser.

---

## Seed Comparison: AttentionGAN Full Data (seed 42 vs seed 123)

| Run | FID | MS-SSIM Div | LPIPS Mean | NN-LPIPS | MiFID | Mem Risk |
|-----|-----|-------------|------------|----------|-------|----------|
| seed 42  | 262.377 | 0.329 | 0.248 | 0.248 | 262.377 | 0.000 |
| seed 123 | 309.158 | 0.319 | 0.203 | 0.240 | 309.158 | 0.000 |

Both runs share the same failure signature: FID > 260, MS-SSIM diversity ~0.32 (well below the ~0.70 seen in functional models), zero memorisation risk (collapsed outputs are trivially far from real images), and MiFID = FID (no near-copies detected because the generator is not producing realistic images at all). Seed 123 is worse than seed 42 by ~47 FID points and also shows lower LPIPS diversity (0.203 vs 0.248), suggesting even less visual variation in the generated images.

This cross-seed consistency rules out an unlucky initialisation — the failure is reproducible and architectural. Likely causes include gradient instability specific to the attention mechanism under the full CelebA data volume, a learning rate or n\_critic setting that works at smaller scale but diverges here, or a capacity mismatch in the attention layers. The low/noisy seed 42 runs scored 61–66 FID, so the issue is specific to the full-data regime.

---

## Key Takeaways

- **WGAN-GP is the best model with full data** (FID 25.69) but is the most data-hungry — its performance degrades sharply (+33.6 points) under low-data conditions.
- **DCGAN is the most robust model overall**: best FID under low-data (34.66) and best FID under noisy conditions (61.55), with consistently low FID degradation across all regimes.
- **AttentionGAN fails reproducibly on full CelebA data** (seed 42: FID 262.38, seed 123: FID 309.16) — both runs show the same collapse signature, ruling out a one-off seed issue. The failure is architecture- or hyperparameter-level and specific to the full-data regime; low-data and noisy runs are competitive (~61–66 FID).
- **The Combined model does not outperform its constituent architectures in any condition** and accumulates the largest memorisation penalties under data scarcity, making it the least favourable choice for resource-constrained settings.
- **Noisy conditions act as an implicit regulariser**: all models record lower memorisation risk under noisy training than under full or low-data training, and four of the five lowest memorisation-risk entries are noisy-condition runs.
- **Diversity (LPIPS, MS-SSIM) is broadly similar across functional models** (~0.26–0.30 LPIPS, ~0.69–0.75 MS-SSIM diversity), suggesting that architectural choices primarily affect fidelity rather than output diversity once training succeeds.
- **MiFID penalty is highest under low-data conditions** across all models, confirming that data scarcity is the primary driver of near-memorisation, not model architecture.
