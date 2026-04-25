# COMP6242 Deep Learning - GAN Experiment Project

This project contains the complete implementation of 12 GAN experiments, designed to study the independent and combined effects of loss function (WGAN-GP) and structural prior (Self-Attention) on GAN training stability.

## Project Structure

```
project/
├── data/                         # Data directory
│   ├── celeba/                  # Source images (e.g. 178x218 CelebA originals)
│   └── eval_real_<ds>_<sz>px_<n>/  # Cached real-image subset for FID, preprocessed to img_size
├── models/
│   ├── __init__.py
│   ├── layers.py             # Self-Attention and other shared components
│   ├── dcgan.py              # M1: Baseline DCGAN
│   ├── wgan_gp.py            # M2: WGAN-GP
│   ├── attention_gan.py      # M3: DCGAN + Self-Attention
│   └── combined.py           # M4: WGAN-GP + Self-Attention
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading module
│   ├── download_data.py      # Data download script (kagglehub)
│   └── visualize.py          # Visualization utilities
├── train.py                  # Unified training entry point
├── evaluate.py               # FID evaluation module
├── requirements.txt          # Dependencies
└── README.md
```

## Environment Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Dataset

```bash
# Anime Face Dataset (default, ~43k images)
python -m utils.download_data

# CelebA Dataset (~202k images)
python -m utils.download_data --dataset celeba
```

A Kaggle account is required for the first run. If you have previously used `kagglehub`, cached credentials will be used.

- Anime faces are downloaded to `data/anime_faces/`
- CelebA is downloaded to `data/celeba/`

To train on CelebA instead of the default, pass `--data_dir data/celeba`.

## Model Description

| Model | Description |
|-------|-------------|
| **M1 - DCGAN** | Baseline, standard DCGAN architecture with BCE Loss |
| **M2 - WGAN-GP** | Uses Wasserstein Loss + Gradient Penalty, D:G=5:1 |
| **M3 - Attention GAN** | DCGAN + Self-Attention module |
| **M4 - Combined** | WGAN-GP + Self-Attention |

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **full_data** | Uses all training data (~50,000 images) |
| **low_data** | Uses 10% of data (~5,000 images), fixed seed sampling |
| **noisy** | Adds Gaussian noise to input (σ=0.1) |

## Usage

### Train a Model

```bash
# M1 + full_data
python train.py --model dcgan --condition full_data --seed 42

# M2 + low_data
python train.py --model wgan_gp --condition low_data --seed 42

# M3 + noisy
python train.py --model attention_gan --condition noisy --seed 42

# M4 + full_data
python train.py --model combined --condition full_data --seed 42
```

### Training Parameters

```bash
python train.py --help

Main parameters:
  --model          Model type (dcgan, wgan_gp, attention_gan, combined)
  --condition      Experimental condition (full_data, low_data, noisy)
  --epochs         Number of training epochs (default: 100)
  --batch_size     Batch size (default: 64)
  --seed           Random seed (default: 42)
  --data_dir       Dataset directory (default: data/anime_faces)
  --exp_dir        Experiment results directory (default: experiments)
  --save_freq      Save frequency (default: every 10 epochs)
  --resume         Resume from checkpoint path
```

### Run All 12 Experiments

```bash
# Create a run script or use the following commands

# DCGAN
python train.py --model dcgan --condition full_data --seed 42
python train.py --model dcgan --condition low_data --seed 42
python train.py --model dcgan --condition noisy --seed 42

# WGAN-GP
python train.py --model wgan_gp --condition full_data --seed 42
python train.py --model wgan_gp --condition low_data --seed 42
python train.py --model wgan_gp --condition noisy --seed 42

# Attention GAN
python train.py --model attention_gan --condition full_data --seed 42
python train.py --model attention_gan --condition low_data --seed 42
python train.py --model attention_gan --condition noisy --seed 42

# Combined
python train.py --model combined --condition full_data --seed 42
python train.py --model combined --condition low_data --seed 42
python train.py --model combined --condition noisy --seed 42
```

### Evaluation (FID + KID + IS + Diversity + MiFID Proxy)

```bash
# Default evaluation (uses all enabled metrics)
python evaluate.py --exp_dir experiments/dcgan_full_data_seed42 --num_samples 10000

# For CelebA experiments
python evaluate.py --exp_dir experiments/dcgan_celeba_full_data_seed42 --data_dir data/celeba --num_samples 10000

# Fast re-evaluation when fid_samples already exist
python evaluate.py \
  --exp_dir experiments/dcgan_celeba_full_data_seed42 \
  --data_dir data/celeba \
  --num_samples 10000 \
  --reuse_fake_samples

# Control new metrics sampling cost
python evaluate.py \
  --exp_dir experiments/dcgan_celeba_full_data_seed42 \
  --data_dir data/celeba \
  --num_samples 10000 \
  --reuse_fake_samples \
  --diversity_pairs 128 \
  --mifid_fake_probe_samples 32 \
  --mifid_real_ref_samples 256

# Skip optional metric groups (if needed)
python evaluate.py --exp_dir experiments/dcgan_celeba_full_data_seed42 --skip_diversity_metrics
python evaluate.py --exp_dir experiments/dcgan_celeba_full_data_seed42 --skip_mifid
python evaluate.py --exp_dir experiments/dcgan_celeba_full_data_seed42 --skip_extra_metrics
```

Evaluation results will be saved at `experiments/<exp_name>/fid_results.json`.

Output fields in `fid_results.json`, grouped by category:

- **Identity / config**: `model`, `condition`, `num_samples`, `fid_method` (`pytorch-fid` or `custom-fallback`)
- **Quality / distribution alignment**: `fid_score`, `kid_mean`, `kid_std`, `inception_score_mean`, `inception_score_std`
- **Precision / Recall (manifold split)**: `precision`, `recall`
- **Diversity**: `ms_ssim_mean`, `ms_ssim_diversity`, `lpips_diversity_mean`, `lpips_diversity_std`, `diversity_pairs`
- **Anti-memorization (MiFID proxy)**: `nn_lpips_mean`, `nn_lpips_p05`, `mifid_proxy`, `memorization_risk`, `mifid_tau`, `mifid_fake_probe_samples`, `mifid_real_ref_samples`

### Run all 12 evaluations

```bash
for m in dcgan wgan_gp attention_gan combined; do
  for c in full_data low_data noisy; do
    python evaluate.py \
      --exp_dir experiments/${m}_celeba_${c}_seed42 \
      --data_dir data/celeba \
      --num_samples 10000 \
      --reuse_fake_samples \
      --skip_extra_metrics
  done
done
```

The first run builds the resolution-matched real-image cache (~1 min); the remaining 11 reuse it. `--skip_extra_metrics` skips `torch-fidelity` (KID / IS / Precision / Recall) — recommended on Apple Silicon, where `torch-fidelity` has no MPS backend and falls back to CPU.

### Evaluation methodology

FID is computed via [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (primary path) with a torchvision-Inception fallback for environments where `pytorch-fid` is unavailable. The primary path is the canonical FID implementation used by the GAN literature; both paths use 2048-d Inception features and the standard Fréchet distance between the two Gaussian fits.

To make the real and fake feature distributions comparable, real images are preprocessed to the generator's output resolution (`img_size × img_size`, default 64) using the same deterministic pipeline applied during training: `Resize(img_size) + CenterCrop(img_size)` (no flip / no noise; see `utils/data_loader.get_transforms`). This is required for FID correctness — `pytorch-fid` loads PNGs from disk and stretches them to 299×299 inside its Inception wrapper, so feeding raw 178×218 CelebA originals while the generator outputs 64×64 produces a resolution asymmetry that contaminates the feature distributions and inflates FID by ~3–5×.

Preprocessed real images are cached at `data/eval_real_{dataset}_{img_size}px_{N}/` with a `_source.txt` marker recording the source path, `img_size`, and preprocessing pipeline. The cache is automatically rebuilt when the marker doesn't match (e.g. switching datasets, switching `img_size`, or upgrading from a version of `evaluate.py` that pre-dates this preprocessing).

Generated samples used for FID are saved at `experiments/<exp_name>/fid_samples/` and can be reused across re-evaluations with `--reuse_fake_samples`. Sample generation is seeded (`torch.manual_seed(0)`, plus `torch.mps.manual_seed(0)` when running on Apple Silicon) so the same checkpoint always produces the same FID samples.

## Viewing Results

### TensorBoard

```bash
tensorboard --logdir experiments/<exp_name>/logs
```

### Experiment Directory Structure

After training, each experiment directory contains:

```
experiments/<model>_<condition>_seed<seed>/
├── config.json              # Experiment config
├── generator_final.pt       # Final generator
├── discriminator_final.pt   # Final discriminator
├── loss_curve.png           # Loss curve plot
├── loss_stats.json          # Loss statistics
├── fid_results.json         # FID evaluation results
├── samples/                 # Generated samples (every 10 epochs)
│   ├── samples_epoch_10.png
│   ├── samples_epoch_20.png
│   └── ...
├── checkpoints/             # Checkpoints
│   └── checkpoint_epoch_<N>.pt
└── logs/                    # TensorBoard logs
```

## Model Architecture Details

### Generator

- Input: z ∈ R^100 (latent vector), reshaped to [B, z_dim, 1, 1]
- Architecture: 5-layer ConvTranspose2d (1→4→8→16→32→64)
- Each intermediate layer: BatchNorm + ReLU
- Output layer: Tanh, image [B, 3, 64, 64], range [-1, 1]

### Discriminator

- Input: image [B, 3, 64, 64]
- Architecture: 4-layer Conv2d
- Each layer: BatchNorm + LeakyReLU(0.2)
- Output: real/fake probability [B, 1] (DCGAN) or Wasserstein distance estimate (WGAN-GP)

### Self-Attention

- Position: after the second-to-last layer in G and D
- Computation: Q, K, V obtained through 1×1 convolutions
- Output: gamma * attention(V) + x (gamma is learnable)

## Training Configuration

### Common Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 64 |
| Image Size | 64×64 |
| z_dim | 100 |
| Seed | 42 |

### DCGAN / Attention GAN

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Learning Rate | 2e-4 |
| Loss | BCE |

### WGAN-GP / Combined

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (β1=0.0, β2=0.9) |
| Learning Rate | 1e-4 |
| Loss | Wasserstein + GP (λ=10) |
| D:G Ratio | 5:1 |

## Data Processing

- Normalization: [-1, 1]
- Data augmentation: only full_data condition uses random horizontal flip
- Low Data: samples 10% of data using fixed seed (42)
- Noisy: adds N(0, 0.1) Gaussian noise

## Evaluation Metrics

- **Quality / distribution alignment**
  - **FID** (lower is better)
  - **KID** (lower is better)
  - **IS** (higher is better; less informative for modern GAN comparison than FID/KID)
- **Precision / Recall (distribution support split)**
  - **Precision**: fidelity to real manifold (higher is better)
  - **Recall**: coverage/diversity over real manifold (higher is better)
- **Diversity metrics**
  - **MS-SSIM mean**: lower indicates more diversity
  - **MS-SSIM diversity = 1 - MS-SSIM**: higher indicates more diversity
  - **LPIPS diversity mean/std**: higher usually indicates richer perceptual variation
- **Anti-memorization metrics**
  - **NN LPIPS mean/p05**: nearest-neighbor perceptual distance from generated to real images
  - **MiFID proxy**: FID with a nearest-neighbor penalty to flag potential memorization risk (lower is better)
  - **memorization_risk**: normalized risk indicator derived from NN LPIPS (higher means riskier)
- **Training stability**
  - Loss curve statistics (`g_std`, `d_std`)
- **Visualization**
  - Generated sample grids and per-experiment sample images

Notes on interpretation:

- No single metric should be used as the sole conclusion for GAN performance.
- Prefer a joint reading of **FID/KID + Precision/Recall + Diversity (MS-SSIM, LPIPS)**.
- Prefer the `pytorch-fid` FID values (`fid_method = "pytorch-fid"`); the custom torchvision-Inception fallback is provided for portability and produces values on a *different* feature-space scale that are not directly comparable to the `pytorch-fid` ones (see Evaluation methodology).
- `mifid_proxy` in this repo is a practical anti-memorization proxy built from NN-LPIPS; use it as a risk signal rather than an absolute theorem-level proof.

## References

- Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", ICLR 2016
- Arjovsky et al., "Wasserstein GAN", ICML 2017
- Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017
- Zhang et al., "Self-Attention Generative Adversarial Networks", ICML 2019
- Z. Wang, E. P. Simoncelli, and A. C. Bovik, "Multi-scale structural similarity for image quality assessment," in Proc. 37th Asilomar Conf. Signals, Systems and Computers, 2003, pp. 1398-1402.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," in Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586-595.
- R. Webster, J. Rabin, L. Simon, and F. Jurie, "Detecting Overfitting of Deep Generative Networks via Latent Recovery," in Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11273-11282.
- G. J. J. van den Burg and C. K. I. Williams, "On Memorization in Probabilistic Deep Generative Models," in Advances in Neural Information Processing Systems (NeurIPS), 2021.