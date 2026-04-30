#!/bin/bash
set -e

# Train
python train.py --model attention_gan_g_bounded  --condition full_data --data_dir data/celeba       --seed 42
python train.py --model attention_gan_gd_bounded --condition full_data --data_dir data/celeba       --seed 42
python train.py --model attention_gan_g_bounded  --condition full_data --data_dir data/anime_faces  --seed 42
python train.py --model attention_gan_gd_bounded --condition full_data --data_dir data/anime_faces  --seed 42

# Evaluate
python evaluate.py --exp_dir experiments/attention_gan_g_bounded_celeba_full_data_seed42      --data_dir data/celeba
python evaluate.py --exp_dir experiments/attention_gan_gd_bounded_celeba_full_data_seed42     --data_dir data/celeba
python evaluate.py --exp_dir experiments/attention_gan_g_bounded_anime_faces_full_data_seed42  --data_dir data/anime_faces
python evaluate.py --exp_dir experiments/attention_gan_gd_bounded_anime_faces_full_data_seed42 --data_dir data/anime_faces

# Report
python generate_eval_report.py --exp_root experiments --out_dir reports
