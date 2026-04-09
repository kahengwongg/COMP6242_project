# COMP6242 Deep Learning - GAN Experiment Project

This project contains the complete implementation of 12 GAN experiments, designed to study the independent and combined effects of loss function (WGAN-GP) and structural prior (Self-Attention) on GAN training stability.

## Project Structure

```
project/
├── data/                         # Data directory
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

### FID Evaluation

```bash
python evaluate.py --exp_dir experiments/dcgan_full_data_seed42 --num_samples 5000

# For CelebA experiments
python evaluate.py --exp_dir experiments/dcgan_full_data_seed42 --data_dir data/celeba --num_samples 5000
```

Evaluation results will be saved at `experiments/<exp_name>/fid_results.json`.

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

- **Primary metric**: FID (Fréchet Inception Distance)
- **Secondary metric**: Loss curve standard deviation (training stability)
- **Visualization**: generated samples at fixed epochs (25, 50, 75, 100)

## References

- Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", ICLR 2016
- Arjovsky et al., "Wasserstein GAN", ICML 2017
- Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017
- Zhang et al., "Self-Attention Generative Adversarial Networks", ICML 2019