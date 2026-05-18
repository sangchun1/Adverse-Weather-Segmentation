#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_baseline.sh
#   bash scripts/run_baseline.sh configs/baseline.yaml
#   bash scripts/run_baseline.sh configs/baseline.yaml fog
#   CONDITION=fog bash scripts/run_baseline.sh
#
# If CONDITION is empty, all weather conditions are used.
# If CONDITION is set to fog/rain/snow/night, train/evaluate/visualize use only that condition.
#
# Output structure:
#   outputs/logs/baseline_YYYYMMDD_HHMMSS.log
#   outputs/results/baseline/
#   outputs/visualizations/baseline/
#
# If condition is given:
#   outputs/logs/baseline_fog_YYYYMMDD_HHMMSS.log
#   outputs/results/baseline_fog/
#   outputs/visualizations/baseline_fog/

CONFIG_PATH="${1:-configs/baseline.yaml}"
CONDITION="${2:-${CONDITION:-}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -n "${CONDITION}" ]]; then
  EXPERIMENT_NAME="baseline_${CONDITION}"
else
  EXPERIMENT_NAME="baseline"
fi

LOG_RUN_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"

LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/${LOG_RUN_NAME}.log"

RESULT_DIR="outputs/results/${EXPERIMENT_NAME}"
VIS_DIR="outputs/visualizations/${EXPERIMENT_NAME}"

EVAL_SPLIT="${EVAL_SPLIT:-val}"
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-5}"
VIS_SEED="${VIS_SEED:-42}"

CHECKPOINT_PATH="$(
python - <<PY
import yaml
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ckpt_cfg = cfg.get("checkpoint", {})
save_dir = ckpt_cfg.get("save_dir", "outputs/checkpoints/baseline")
save_best_name = ckpt_cfg.get("save_best_name", "best_miou.pth")

print(str(Path(save_dir) / save_best_name))
PY
)"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"
mkdir -p "${VIS_DIR}"
mkdir -p "outputs/checkpoints"
mkdir -p "outputs"

export CONFIG_PATH
export CONDITION
export CHECKPOINT_PATH
export EVAL_SPLIT
export RESULT_DIR
export VIS_DIR
export SAMPLES_PER_CONDITION
export VIS_SEED

echo "Starting baseline pipeline with nohup..."
echo "Config: ${CONFIG_PATH}"
echo "Experiment: ${EXPERIMENT_NAME}"

if [[ -n "${CONDITION}" ]]; then
  echo "Condition: ${CONDITION}"
else
  echo "Condition: all"
fi

echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Evaluation split: ${EVAL_SPLIT}"
echo "Result dir: ${RESULT_DIR}"
echo "Visualization dir: ${VIS_DIR}"
echo "Samples per condition: ${SAMPLES_PER_CONDITION}"
echo "Visualization seed: ${VIS_SEED}"

nohup bash -c '
set -euo pipefail

CONDITION_ARGS=()
if [[ -n "${CONDITION}" ]]; then
  CONDITION_ARGS=(--condition "${CONDITION}")
fi

echo "============================================================"
echo "[1/3] Training baseline model"
echo "============================================================"

python -m awseg.train \
  --config "${CONFIG_PATH}" \
  --result-dir "${RESULT_DIR}" \
  "${CONDITION_ARGS[@]}"

echo ""
echo "============================================================"
echo "[2/3] Evaluating best checkpoint"
echo "============================================================"

python -m awseg.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split "${EVAL_SPLIT}" \
  --result-dir "${RESULT_DIR}" \
  "${CONDITION_ARGS[@]}"

echo ""
echo "============================================================"
echo "[3/3] Saving visualization results"
echo "============================================================"

python -m awseg.visualize \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split "${EVAL_SPLIT}" \
  --output-dir "${VIS_DIR}" \
  --samples-per-condition "${SAMPLES_PER_CONDITION}" \
  --shuffle \
  --seed "${VIS_SEED}" \
  "${CONDITION_ARGS[@]}"

echo ""
echo "============================================================"
echo "Baseline pipeline finished successfully."
echo "Best checkpoint: ${CHECKPOINT_PATH}"
echo "Result dir: ${RESULT_DIR}"
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
