#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_proposed_new.sh
#   bash scripts/run_proposed_new.sh configs/proposed_new.yaml
#   bash scripts/run_proposed_new.sh configs/proposed_new.yaml fog
#   CONDITION=fog bash scripts/run_proposed_new.sh
#
# If CONDITION is empty, all weather conditions are used.
# If CONDITION is set to fog/rain/snow/night, train/evaluate/visualize/analyze use only that condition.
#
# Output structure:
#   outputs/logs/proposed_new_YYYYMMDD_HHMMSS.log
#   outputs/results/proposed_new/
#   outputs/visualizations/proposed_new/
#   outputs/analysis/proposed_new/
#
# If condition is given:
#   outputs/logs/proposed_new_fog_YYYYMMDD_HHMMSS.log
#   outputs/results/proposed_new_fog/
#   outputs/visualizations/proposed_new_fog/
#   outputs/analysis/proposed_new_fog/

CONFIG_PATH="${1:-configs/proposed_new.yaml}"
CONDITION="${2:-${CONDITION:-}}"
DEVICE="${3:-${DEVICE:-cuda:0}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -n "${CONDITION}" ]]; then
  EXPERIMENT_NAME="proposed_new_${CONDITION}"
else
  EXPERIMENT_NAME="proposed_new"
fi

LOG_RUN_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"

LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/${LOG_RUN_NAME}.log"

RESULT_DIR="outputs/results/${EXPERIMENT_NAME}"
VIS_DIR="outputs/visualizations/${EXPERIMENT_NAME}"
ANALYSIS_DIR="outputs/analysis/${EXPERIMENT_NAME}"

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
save_dir = ckpt_cfg.get("save_dir", "outputs/checkpoints/proposed_new")
save_best_name = ckpt_cfg.get("save_best_name", "best_miou.pth")

print(str(Path(save_dir) / save_best_name))
PY
)"

CHECKPOINT_DIR="$(dirname "${CHECKPOINT_PATH}")"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"
mkdir -p "${VIS_DIR}"
mkdir -p "${ANALYSIS_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "outputs"

export CONFIG_PATH
export CONDITION
export DEVICE
export CHECKPOINT_PATH
export EVAL_SPLIT
export RESULT_DIR
export VIS_DIR
export ANALYSIS_DIR
export SAMPLES_PER_CONDITION
export VIS_SEED

echo "Starting proposed_new pipeline with nohup..."
echo "Config: ${CONFIG_PATH}"
echo "Experiment: ${EXPERIMENT_NAME}"

if [[ -n "${CONDITION}" ]]; then
  echo "Condition: ${CONDITION}"
else
  echo "Condition: all"
fi

echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Device: ${DEVICE}"
echo "Evaluation split: ${EVAL_SPLIT}"
echo "Result dir: ${RESULT_DIR}"
echo "Visualization dir: ${VIS_DIR}"
echo "Analysis dir: ${ANALYSIS_DIR}"
echo "Samples per condition: ${SAMPLES_PER_CONDITION}"
echo "Visualization seed: ${VIS_SEED}"

nohup bash -c '
set -euo pipefail

CONDITION_ARGS=()
if [[ -n "${CONDITION}" ]]; then
  CONDITION_ARGS=(--condition "${CONDITION}")
fi

DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

ANALYZE_CONDITION="${CONDITION}"
if [[ -z "${ANALYZE_CONDITION}" ]]; then
  ANALYZE_CONDITION="none"
fi

echo "============================================================"
echo "[1/5] Training proposed_new model"
echo "============================================================"

python -m awseg.train \
  --config "${CONFIG_PATH}" \
  --result-dir "${RESULT_DIR}" \
  "${DEVICE_ARGS[@]}" \
  "${CONDITION_ARGS[@]}"

echo ""
echo "============================================================"
echo "[2/5] Evaluating best checkpoint"
echo "============================================================"

python -m awseg.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split "${EVAL_SPLIT}" \
  --result-dir "${RESULT_DIR}" \
  "${DEVICE_ARGS[@]}" \
  "${CONDITION_ARGS[@]}"

echo ""
echo "============================================================"
echo "[3/5] Saving visualization results"
echo "============================================================"

python -m awseg.visualize \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split "${EVAL_SPLIT}" \
  --output-dir "${VIS_DIR}" \
  "${DEVICE_ARGS[@]}" \
  --samples-per-condition "${SAMPLES_PER_CONDITION}" \
  --shuffle \
  --seed "${VIS_SEED}" \
  "${CONDITION_ARGS[@]}"

echo ""
echo "============================================================"
echo "[4/5] Running error analysis"
echo "============================================================"

python scripts/analyze_errors.py \
  --group proposed_new \
  --condition "${ANALYZE_CONDITION}" \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output-dir "${ANALYSIS_DIR}" \
  "${DEVICE_ARGS[@]}"

echo ""
echo "============================================================"
echo "[5/5] Plotting proposed_new results"
echo "============================================================"

python scripts/plot_results.py \
  --group proposed_new \
  --output-dir "${VIS_DIR}/plots"

echo ""
echo "============================================================"
echo "Proposed_new pipeline finished successfully."
echo "Best checkpoint: ${CHECKPOINT_PATH}"
echo "Result dir: ${RESULT_DIR}"
echo "Visualization dir: ${VIS_DIR}"
echo "Analysis dir: ${ANALYSIS_DIR}"
echo "============================================================"
' > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Pipeline started."
echo "PID: ${PID}"
echo "To monitor logs:"
echo "tail -f ${LOG_FILE}"
echo ""
echo "After training finishes, evaluation, visualization, analysis, and plotting will run automatically."