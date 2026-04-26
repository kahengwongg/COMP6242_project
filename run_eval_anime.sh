#!/bin/bash
# Evaluate all 12 experiments in experiments/anime_train_100ep/
set -e

EXP_ROOT="experiments/anime_train_100ep"
DATA_DIR="data/anime_faces"
NUM_SAMPLES=5000

MODELS=("dcgan" "wgan_gp" "attention_gan" "combined")
CONDITIONS=("full_data" "low_data" "noisy")

cd /workspace/COMP6242_project

for model in "${MODELS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        exp_name="${model}_anime_faces_${condition}_seed42"
        exp_dir="${EXP_ROOT}/${exp_name}"
        echo "============================================================"
        echo "Evaluating: ${exp_name}"
        echo "============================================================"
        python3 evaluate.py \
            --exp_dir "${exp_dir}" \
            --data_dir "${DATA_DIR}" \
            --num_samples ${NUM_SAMPLES} \
            --batch_size 64 \
            --device auto
    done
done

echo "All evaluations complete."
