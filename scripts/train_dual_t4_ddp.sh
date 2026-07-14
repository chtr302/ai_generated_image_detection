#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-}"
HF_DATASET="${HF_DATASET:-Rajarshi-Roy-research/Defactify_Image_Dataset}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
HF_NO_STREAMING="${HF_NO_STREAMING:-0}"
HF_SHUFFLE_BUFFER="${HF_SHUFFLE_BUFFER:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ai_detector}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NEC="${NEC:-10}"
AMP="${AMP:-fp16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"

ARGS=(
  --nproc_per_node=2
  -m src.model.train
  --output-dir "$OUTPUT_DIR"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --grad-accum "$GRAD_ACCUM"
  --nec "$NEC"
  --amp "$AMP"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "$DATA_ROOT" ]]; then
  ARGS+=(--data-root "$DATA_ROOT")
else
  ARGS+=(--hf-dataset "$HF_DATASET" --hf-shuffle-buffer "$HF_SHUFFLE_BUFFER")
  if [[ -n "$HF_CACHE_DIR" ]]; then
    ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
  fi
  if [[ "$HF_NO_STREAMING" == "1" ]]; then
    ARGS+=(--hf-no-streaming)
  fi
fi

if [[ -n "${RESUME:-}" ]]; then
  ARGS+=(--resume "$RESUME")
fi
if [[ "$MAX_TRAIN_STEPS" != "0" ]]; then
  ARGS+=(--max-train-steps "$MAX_TRAIN_STEPS")
fi

torchrun "${ARGS[@]}"
