#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/baseline.yaml}"

RUN_NAME="baseline_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

CHECKPOINT_PATH="checkpoints/baseline/best.pth"
EVAL_SPLIT="val"
VIS_DIR="outputs/visualizations/${RUN_NAME}"
NUM_VIS_SAMPLES="${NUM_VIS_SAMPLES:-20}"

mkdir -p "${LOG_DIR}"
mkdir -p "${VIS_DIR}"

export CONFIG_PATH
export CHECKPOINT_PATH
export EVAL_SPLIT
export VIS_DIR
export NUM_VIS_SAMPLES

echo "Starting baseline pipeline with nohup..."
echo "Config: ${CONFIG_PATH}"
echo "Log file: ${LOG_FILE}"
echo "Visualization dir: ${VIS_DIR}"

nohup bash -c '
set -euo pipefail

echo "============================================================"
echo "[1/3] Training baseline model"
echo "============================================================"
python -m awseg.train \
  --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "[2/3] Evaluating best checkpoint"
echo "============================================================"
python -m awseg.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split "${EVAL_SPLIT}"

echo ""
echo "============================================================"
echo "[3/3] Saving visualization results"
echo "============================================================"
python -m awseg.visualize \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split "${EVAL_SPLIT}" \
  --output-dir "${VIS_DIR}" \
  --num-samples "${NUM_VIS_SAMPLES}"

echo ""
echo "============================================================"
echo "Baseline pipeline finished successfully."
echo "Best checkpoint: ${CHECKPOINT_PATH}"
echo "Visualization dir: ${VIS_DIR}"
echo "============================================================"
' > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Pipeline started."
echo "PID: ${PID}"
echo "To monitor logs:"
echo "tail -f ${LOG_FILE}"
echo ""
echo "After training finishes, evaluation and visualization will run automatically."
