# COMP6242 GAN Experiment Project

This repository contains the code for the 12 main CelebA GAN experiments used in the project report.

## Project Structure

```
project/
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
│   └── visualize.py          # Visualisation utilities
├── train.py                  # Unified training entry point
├── evaluate.py               # Evaluation script
├── scripts/                  # Utility scripts
├── requirements.txt          # Dependencies
└── README.md
```

## Environment Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python -m utils.download_data --dataset celeba
```

A Kaggle account is required for the first run. If you have previously used `kagglehub`, cached credentials will be used.
CelebA is downloaded to `data/celeba/`.

## Model Description

| Model | Description |
|-------|-------------|
| **M1 - DCGAN** | Baseline DCGAN with BCE loss |
| **M2 - WGAN-GP** | Uses Wasserstein Loss + Gradient Penalty, D:G=5:1 |
| **M3 - Attention GAN** | DCGAN + Self-Attention module |
| **M4 - Combined** | WGAN-GP + Self-Attention |

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| **full_data** | Uses all images in the selected dataset directory |
| **low_data** | Uses 10% of the selected dataset, fixed seed sampling |
| **noisy** | Adds Gaussian noise to input (σ=0.1) |

## Usage

### Train a Model

```bash
# M1 + full_data on CelebA
python train.py --model dcgan --condition full_data --seed 42 --data_dir data/celeba

# M2 + low_data on CelebA
python train.py --model wgan_gp --condition low_data --seed 42 --data_dir data/celeba

# M3 + noisy on CelebA
python train.py --model attention_gan --condition noisy --seed 42 --data_dir data/celeba

# M4 + full_data on CelebA
python train.py --model combined --condition full_data --seed 42 --data_dir data/celeba
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
  --data_dir       Dataset directory (default: data/celeba)
  --exp_dir        Experiment results directory (default: experiments)
  --loss_type      Optional loss override for BCE models (bce, bce_logits)
  --save_freq      Save frequency (default: every 10 epochs)
  --resume         Resume from checkpoint path
```

### Run All 12 Experiments

```bash
for m in dcgan wgan_gp attention_gan combined; do
  for c in full_data low_data noisy; do
    python train.py \
      --model "$m" \
      --condition "$c" \
      --seed 42 \
      --data_dir data/celeba
  done
done
```

The main 12 experiments use the default loss settings. For `dcgan` and `attention_gan`, the default is `--loss_type bce`; for `wgan_gp` and `combined`, the WGAN-GP objective is used.

### Evaluation

```bash
python evaluate.py \
  --exp_dir experiments/dcgan_celeba_full_data_seed42 \
  --data_dir data/celeba \
  --num_samples 10000 \
  --reuse_fake_samples
```

Evaluation results will be saved at `experiments/<exp_name>/fid_results.json`.

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

Use `--reuse_fake_samples` when generated FID samples already exist. Use `--skip_extra_metrics` if only FID is needed or optional metric dependencies are unavailable.

## Outputs

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
| Loss | BCE by default; `attention_gan` can also be run with `--loss_type bce_logits` |

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
