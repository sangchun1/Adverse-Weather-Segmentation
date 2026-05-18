#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_night.sh
#   bash scripts/run_night.sh night_gamma
#   bash scripts/run_night.sh night_sci
#
# Note:
#   This script always uses configs/night.yaml.
#   Edit configs/night.yaml before running each experiment.

CONFIG_PATH="configs/night.yaml"
EXPERIMENT_NAME="${1:-night}"

CONDITION="night"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"

LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

EVAL_SPLIT="${EVAL_SPLIT:-val}"
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-5}"
VIS_SEED="${VIS_SEED:-42}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"

RESULT_DIR="outputs/results/${EXPERIMENT_NAME}"
VIS_DIR="outputs/visualizations/${EXPERIMENT_NAME}"

mkdir -p "${LOG_DIR}"
mkdir -p "${VIS_DIR}"
mkdir -p "outputs/checkpoints"
mkdir -p "outputs"

CHECKPOINT_PATH="$(
python - <<PY
import yaml
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ckpt_cfg = cfg.get("checkpoint", {})
save_dir = ckpt_cfg.get("save_dir", "outputs/checkpoints/night")
save_best_name = ckpt_cfg.get("save_best_name", "best_miou.pth")

print(str(Path(save_dir) / save_best_name))
PY
)"

EVAL_OUTPUT_DIR="$(
python - <<PY
import yaml
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

eval_cfg = cfg.get("evaluate", {})
output_dir = eval_cfg.get("output_dir", "outputs/evaluations/night")

print(str(Path(output_dir)))
PY
)"

mkdir -p "${EVAL_OUTPUT_DIR}"

export CONFIG_PATH
export CONDITION
export CHECKPOINT_PATH
export EVAL_SPLIT
export VIS_DIR
export SAMPLES_PER_CONDITION
export VIS_SEED
export SAVE_PREDICTIONS
export EVAL_OUTPUT_DIR
export RESULT_DIR

echo "Starting night pipeline with nohup..."
echo "Config: ${CONFIG_PATH}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Condition: ${CONDITION}"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Evaluation split: ${EVAL_SPLIT}"
echo "Result dir: ${RESULT_DIR}"
echo "Visualization dir: ${VIS_DIR}"
echo "Samples per condition: ${SAMPLES_PER_CONDITION}"
echo "Visualization seed: ${VIS_SEED}"
echo "Save predictions: ${SAVE_PREDICTIONS}"

nohup bash -c '
set -euo pipefail

CONDITION_ARGS=(--condition "${CONDITION}")

echo "============================================================"
echo "[1/3] Training night model"
echo "============================================================"

python -m awseg.train \
  --config "${CONFIG_PATH}" \
  --result-dir "${RESULT_DIR}" \
  "${CONDITION_ARGS[@]}"

echo ""
echo "============================================================"
echo "[2/3] Evaluating best checkpoint"
echo "============================================================"

EVAL_ARGS=()
if [[ "${SAVE_PREDICTIONS}" == "true" ]]; then
  EVAL_ARGS+=(--save-predictions)
fi

python -m awseg.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --split "${EVAL_SPLIT}" \
  --result-dir "${RESULT_DIR}" \
  "${EVAL_ARGS[@]}" \
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
echo "Night pipeline finished successfully."
echo "Best checkpoint: ${CHECKPOINT_PATH}"
echo "Evaluation output dir: ${EVAL_OUTPUT_DIR}"
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