#!/bin/bash
# Batch run 12 experiments
# Usage: bash run_all_experiments.sh

MODELS=("dcgan" "wgan_gp" "attention_gan" "combined")
CONDITIONS=("full_data" "low_data" "noisy")
SEED=42
DATA_DIR="data/anime_faces"
DATASET=$(basename "$DATA_DIR")

echo "========================================"
echo "COMP6242 GAN Batch Experiment Script"
echo "Total: 12 experiments"
echo "========================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Counter
TOTAL=0
SUCCESS=0

for MODEL in "${MODELS[@]}"; do
    for CONDITION in "${CONDITIONS[@]}"; do
        TOTAL=$((TOTAL + 1))
        EXP_NAME="${MODEL}_${DATASET}_${CONDITION}_seed${SEED}"
        
        echo "----------------------------------------"
        echo "Experiment $TOTAL/12: $EXP_NAME"
        echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "----------------------------------------"
        
        # Run training
        if python train.py --model "$MODEL" --condition "$CONDITION" --seed "$SEED" --data_dir "$DATA_DIR"; then
            SUCCESS=$((SUCCESS + 1))
            echo "Done: $EXP_NAME"
        else
            echo "Failed: $EXP_NAME"
        fi
        
        echo ""
    done
done

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "========================================"
echo "All experiments complete!"
echo "Success: $SUCCESS/$TOTAL"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "========================================"
echo ""

# Ask whether to run FID evaluation
read -p "Run FID evaluation? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting FID evaluation..."
    echo ""
    
    for MODEL in "${MODELS[@]}"; do
        for CONDITION in "${CONDITIONS[@]}"; do
            EXP_DIR="experiments/${MODEL}_${DATASET}_${CONDITION}_seed${SEED}"
            
            if [ -d "$EXP_DIR" ]; then
                echo "Evaluating: $EXP_DIR"
                python evaluate.py --exp_dir "$EXP_DIR" --num_samples 5000 --device auto
                echo ""
            fi
        done
    done
    
    echo "FID evaluation complete!"
fi
