#!/usr/bin/env bash
set -euo pipefail

# Colab single-GPU training entrypoint.
# Run from the repository root after mounting Drive or cloning the repo.

DATA_ROOT="${DATA_ROOT:-/content/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/ai_detector_outputs}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NEC="${NEC:-10}"
AMP="${AMP:-fp16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RESUME="${RESUME:-}"

ARGS=(
  -m src.model.train
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --grad-accum "$GRAD_ACCUM"
  --nec "$NEC"
  --amp "$AMP"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

python "${ARGS[@]}"
