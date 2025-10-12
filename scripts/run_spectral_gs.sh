#!/bin/bash
# scripts/run_spectral_gs.sh
# Train Spectral-GS on COLMAP dataset

DATA_DIR=${1:-"data/campus_scene"}
RESULT_DIR=${2:-"results/spectral_gs"}
DATA_FACTOR=${3:-1}
MAX_STEPS=${4:-10000}
SPECTRAL_THRESHOLD=${5:-0.5}

echo "============================================"
echo "Training Spectral-GS"
echo "============================================"
echo "Data:               $DATA_DIR"
echo "Output:             $RESULT_DIR"
echo "Data factor:        $DATA_FACTOR"
echo "Max steps:          $MAX_STEPS"
echo "Spectral threshold: $SPECTRAL_THRESHOLD"
echo "============================================"
echo ""

python scripts/train_spectral_gs.py \
  --data_dir "$DATA_DIR" \
  --result_dir "$RESULT_DIR" \
  --data_factor "$DATA_FACTOR" \
  --max_steps "$MAX_STEPS" \
  --spectral_threshold "$SPECTRAL_THRESHOLD" \
  --enable_spectral_splitting \
  --enable_filtering \
  --save_ply \
  --verbose

echo ""
echo "============================================"
echo "Training complete!"
echo "Results saved to: $RESULT_DIR"
echo "============================================"
