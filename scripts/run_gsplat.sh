#!/bin/bash
# scripts/run_gsplat.sh
# Train Gaussian Splatting on COLMAP dataset using gsplat

DATA_DIR=${1:-"data/campus_scene"}
RESULT_DIR=${2:-"results/baseline"}
DATA_FACTOR=${3:-1}
MAX_STEPS=${4:-10000}

echo "Training Gaussian Splatting..."
echo "Data: $DATA_DIR"
echo "Output: $RESULT_DIR"

python examples/simple_trainer.py default \
  --data_dir "$DATA_DIR" \
  --result_dir "$RESULT_DIR" \
  --data_factor "$DATA_FACTOR" \
  --max_steps "$MAX_STEPS" \
  --save_ply \
  --sh_degree 3 \
  --ssim_lambda 0.2

echo "Training complete! Results saved to: $RESULT_DIR"