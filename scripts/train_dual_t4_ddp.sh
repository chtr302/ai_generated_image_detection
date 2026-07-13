#!/usr/bin/env bash
set -euo pipefail

# Chay tu root cua repo.
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ai_detector}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NEC="${NEC:-10}"
AMP="${AMP:-fp16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

ARGS=(
  --nproc_per_node=2
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

if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "$RESUME")
fi

torchrun "${ARGS[@]}"
