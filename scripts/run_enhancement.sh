#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_enhancement.sh
#   bash scripts/run_enhancement.sh configs/enhance.yaml
#   bash scripts/run_enhancement.sh configs/enhance_night_only.yaml
#   bash scripts/run_enhancement.sh configs/enhance.yaml enhance_gamma
#   bash scripts/run_enhancement.sh configs/enhance_night_only.yaml enhance_night_only_gamma
#
# Notes:
#   - This script is for all-condition enhancement experiments.
#   - It does NOT pass --condition.
#   - Therefore train/evaluate/visualize use all conditions from data/splits.
#   - Enhancement scope is controlled by YAML:
#       apply_conditions: "all"
#       apply_conditions: ["night"]
#   - Edit the selected YAML to switch gamma <-> clahe.

CONFIG_PATH="${1:-configs/enhance.yaml}"
EXPERIMENT_NAME_ARG="${2:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}"
  exit 1
fi

EXPERIMENT_NAME="$(
python - <<PY
import yaml
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
experiment_name_arg = "${EXPERIMENT_NAME_ARG}"

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if experiment_name_arg:
    print(experiment_name_arg)
else:
    ckpt_cfg = cfg.get("checkpoint", {})
    save_dir = ckpt_cfg.get("save_dir", "outputs/checkpoints/enhance")
    print(Path(save_dir).name)
PY
)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_RUN_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"

LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/${LOG_RUN_NAME}.log"

EVAL_SPLIT="${EVAL_SPLIT:-val}"
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-5}"
VIS_SEED="${VIS_SEED:-42}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"

VIS_DIR="outputs/visualizations/${EXPERIMENT_NAME}"

CHECKPOINT_PATH="$(
python - <<PY
import yaml
from pathlib import Path

config_path = Path("${CONFIG_PATH}")

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ckpt_cfg = cfg.get("checkpoint", {})
save_dir = ckpt_cfg.get("save_dir", "outputs/checkpoints/enhance")
save_best_name = ckpt_cfg.get("save_best_name", "best_miou.pth")

print(str(Path(save_dir) / save_best_name))
PY
)"

RESULT_DIR="$(
python - <<PY
import yaml
from pathlib import Path

config_path = Path("${CONFIG_PATH}")

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

eval_cfg = cfg.get("evaluate", {})
output_dir = eval_cfg.get("output_dir", "outputs/results/${EXPERIMENT_NAME}")

print(str(Path(output_dir)))
PY
)"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"
mkdir -p "${VIS_DIR}"
mkdir -p "outputs/checkpoints"
mkdir -p "outputs/results"
mkdir -p "outputs/visualizations"
mkdir -p "outputs"

export CONFIG_PATH
export CHECKPOINT_PATH
export EVAL_SPLIT
export RESULT_DIR
export VIS_DIR
export SAMPLES_PER_CONDITION
export VIS_SEED
export SAVE_PREDICTIONS

echo "Starting enhancement pipeline with nohup..."
echo "Config: ${CONFIG_PATH}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Condition: all"
echo "Log file: ${LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Evaluation split: ${EVAL_SPLIT}"
echo "Result dir: ${RESULT_DIR}"
echo "Visualization dir: ${VIS_DIR}"
echo "Samples per condition: ${SAMPLES_PER_CONDITION}"
echo "Visualization seed: ${VIS_SEED}"
echo "Save predictions: ${SAVE_PREDICTIONS}"
echo ""
echo "Note: This script does NOT pass --condition."
echo "      All train/val/test samples from data/splits are used."
echo "      Enhancement scope is controlled by enhancement.apply_conditions in YAML."

nohup bash -c '
set -euo pipefail

echo "============================================================"
echo "[1/3] Training enhancement model on all conditions"
echo "============================================================"

python -m awseg.train \
  --config "${CONFIG_PATH}" \
  --result-dir "${RESULT_DIR}"

echo ""
echo "============================================================"
echo "[2/3] Evaluating best checkpoint on all validation data"
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
  "${EVAL_ARGS[@]}"

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
  --seed "${VIS_SEED}"

echo ""
echo "============================================================"
echo "Enhancement pipeline finished successfully."
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