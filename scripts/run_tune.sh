#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_tune.sh
#   bash scripts/run_tune.sh 20
#   bash scripts/run_tune.sh 20 configs/proposed.yaml
#   bash scripts/run_tune.sh 20 configs/proposed.yaml night cuda:1
#
# Environment variables:
#   MAX_EPOCHS_PER_TRIAL=20 bash scripts/run_tune.sh
#   OUTPUT_DIR=outputs/tuning/segformer bash scripts/run_tune.sh
#   STUDY_NAME=segformer_optuna bash scripts/run_tune.sh
#   STORAGE=sqlite:///outputs/tuning/segformer/optuna.db bash scripts/run_tune.sh
#   DEVICE=cuda:0 bash scripts/run_tune.sh
#   RUN_EVAL=1 bash scripts/run_tune.sh
#
# Notes:
#   - Default condition is empty, which means all weather conditions are used.
#   - Set CONDITION to fog/rain/snow/night or pass it as the 3rd argument.
#   - PyTorch should be installed separately according to CUDA version.

N_TRIALS="${1:-20}"
CONFIG_PATH="${2:-configs/proposed.yaml}"
CONDITION="${3:-${CONDITION:-}}"

MAX_EPOCHS_PER_TRIAL="${MAX_EPOCHS_PER_TRIAL:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tuning/segformer}"
STUDY_NAME="${STUDY_NAME:-segformer_optuna}"
STORAGE="${STORAGE:-sqlite:///${OUTPUT_DIR}/optuna.db}"
DEVICE="${4:-${DEVICE:-cuda:0}}"
RUN_EVAL="${RUN_EVAL:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/segformer_optuna_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

ARGS=(
  --config "${CONFIG_PATH}"
  --n-trials "${N_TRIALS}"
  --max-epochs-per-trial "${MAX_EPOCHS_PER_TRIAL}"
  --output-dir "${OUTPUT_DIR}"
  --study-name "${STUDY_NAME}"
  --storage "${STORAGE}"
  --resume-study
)

if [[ -n "${CONDITION}" ]]; then
  ARGS+=(--condition "${CONDITION}")
fi

if [[ -n "${DEVICE}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "True" ]]; then
  ARGS+=(--run-eval)
fi

echo "Starting SegFormer tuning..."
echo "Config              : ${CONFIG_PATH}"
echo "Condition           : ${CONDITION:-all}"
echo "Trials              : ${N_TRIALS}"
echo "Max epochs per trial: ${MAX_EPOCHS_PER_TRIAL}"
echo "Output dir          : ${OUTPUT_DIR}"
echo "Study name          : ${STUDY_NAME}"
echo "Storage             : ${STORAGE}"
echo "Device              : ${DEVICE:-config/default}"
echo "Run eval            : ${RUN_EVAL}"
echo "Log file            : ${LOG_FILE}"

python scripts/tune_segformer.py "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"