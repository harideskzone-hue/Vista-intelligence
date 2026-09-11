#!/bin/bash
set -e

# Change to the project root directory
cd "/Users/hariharans/Documents/Vehicle Intelligence/SIH26187"

# Activate the virtual environment
source .venv-vehicle/bin/activate

# Ensure the output directory exists
mkdir -p data/fast_plate_ocr/models/fine_tuned

# Run the training
# We use epochs=100 with early stopping patience of 15 based on validation exact match accuracy
fast-plate-ocr train \
  --model-config-file data/fast_plate_ocr/models/cct_xs_v2_global_model_config.yaml \
  --plate-config-file data/fast_plate_ocr/models/cct_xs_v2_global_plate_config.yaml \
  --annotations data/fast_plate_ocr/train_annotations.csv \
  --val-annotations data/fast_plate_ocr/val_annotations.csv \
  --epochs 100 \
  --batch-size 32 \
  --output-dir data/fast_plate_ocr/models/fine_tuned \
  --weights-path data/fast_plate_ocr/models/cct_xs_v2_global.keras \
  --label-smoothing 0.0 \
  --weight-decay 0.0005 \
  --lr 0.0005 \
  --early-stopping-patience 15
