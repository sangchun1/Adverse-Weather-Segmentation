#!/usr/bin/env bash
set -euo pipefail

# Run final SegFormer Optuna tuning with nohup.
#
# Usage:
#   bash scripts/run_final_tune.sh
#   bash scripts/run_final_tune.sh cuda:1
#
# Optional environment overrides:
#   DEVICE=cuda:1 N_TRIALS=50 MAX_EPOCHS_PER_TRIAL=40 bash scripts/run_final_tune.sh
#   CONDITION=night bash scripts/run_final_tune.sh cuda:0
#   RESUME_STUDY=1 bash scripts/run_final_tune.sh cuda:0
#   DISABLE_WANDB=0 bash scripts/run_final_tune.sh cuda:0
#   RUN_EVAL=1 bash scripts/run_final_tune.sh cuda:0

CONFIG_PATH="${CONFIG_PATH:-configs/final.yaml}"
SCRIPT_PATH="${SCRIPT_PATH:-scripts/final_tune.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tuning/segformer_final_refine}"
STUDY_NAME="${STUDY_NAME:-segformer_final_refine}"
STORAGE="${STORAGE:-sqlite:///${OUTPUT_DIR}/optuna.db}"
N_TRIALS="${N_TRIALS:-50}"
MAX_EPOCHS_PER_TRIAL="${MAX_EPOCHS_PER_TRIAL:-40}"
DEVICE="${1:-${DEVICE:-cuda:0}}"
CONDITION="${CONDITION:-}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
RESUME_STUDY="${RESUME_STUDY:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG_DIR="${OUTPUT_DIR}/nohup_logs"
NOHUP_LOG_PATH="${NOHUP_LOG_DIR}/final_tune_${TIMESTAMP}.log"
PID_PATH="${OUTPUT_DIR}/final_tune.pid"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] Script not found: ${SCRIPT_PATH}" >&2
  echo "        Put final_tune.py at scripts/final_tune.py first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${NOHUP_LOG_DIR}"

CMD=(
  python -u "${SCRIPT_PATH}"
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --storage "${STORAGE}"
  --study-name "${STUDY_NAME}"
  --n-trials "${N_TRIALS}"
  --max-epochs-per-trial "${MAX_EPOCHS_PER_TRIAL}"
  --device "${DEVICE}"
)

if [[ -n "${CONDITION}" ]]; then
  CMD+=(--condition "${CONDITION}")
fi

if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  CMD+=(--run-eval)
fi

if [[ "${RESUME_STUDY}" == "1" ]]; then
  CMD+=(--resume-study)
fi

{
  echo "============================================================"
  echo "Final tuning command"
  echo "============================================================"
  printf '%q ' "${CMD[@]}"
  echo
  echo "============================================================"
  echo "Config      : ${CONFIG_PATH}"
  echo "Output dir  : ${OUTPUT_DIR}"
  echo "Study name  : ${STUDY_NAME}"
  echo "Storage     : ${STORAGE}"
  echo "N trials    : ${N_TRIALS}"
  echo "Epochs/trial: ${MAX_EPOCHS_PER_TRIAL}"
  echo "Device      : ${DEVICE}"
  echo "Condition   : ${CONDITION:-all}"
  echo "W&B disabled: ${DISABLE_WANDB}"
  echo "Run eval    : ${RUN_EVAL}"
  echo "Resume study: ${RESUME_STUDY}"
  echo "Log path    : ${NOHUP_LOG_PATH}"
  echo "PID path    : ${PID_PATH}"
  echo "============================================================"
} | tee "${NOHUP_LOG_PATH}"

nohup "${CMD[@]}" >> "${NOHUP_LOG_PATH}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_PATH}"

echo "Started final tuning with PID ${PID}"
echo "Log: ${NOHUP_LOG_PATH}"
echo "PID: ${PID_PATH}"
echo ""
echo "Check progress:"
echo "  tail -f ${NOHUP_LOG_PATH}"
echo ""
echo "Check process:"
echo "  ps -p ${PID} -f"
