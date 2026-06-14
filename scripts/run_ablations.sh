#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_ablations.sh
#   bash scripts/run_ablations.sh cuda:0
#   DEVICE=cuda:1 bash scripts/run_ablations.sh
#   EXPERIMENTS="01_baseline_segformer 02_ce_tversky" bash scripts/run_ablations.sh cuda:0
#
# This script runs final ablation experiments sequentially in the background.
#
# Output structure:
#   outputs/logs/ablations_YYYYMMDD_HHMMSS.log
#   outputs/results/ablations/<experiment>/
#   outputs/visualizations/ablations/<experiment>/
#   outputs/analysis/ablations/<experiment>/
#   outputs/checkpoints/ablations/<experiment>/
#   outputs/tmp_configs/ablations/<experiment>.yaml

DEVICE="${1:-${DEVICE:-cuda:0}}"
CONFIG_DIR="${CONFIG_DIR:-configs/ablations}"
GROUP="${GROUP:-ablations}"
EXPERIMENTS="${EXPERIMENTS:-}"

EVAL_SPLIT="${EVAL_SPLIT:-val}"
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-5}"
VIS_SEED="${VIS_SEED:-42}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/${GROUP}_${TIMESTAMP}.log"
TMP_CONFIG_DIR="outputs/tmp_configs/${GROUP}"

mkdir -p "${LOG_DIR}" "${TMP_CONFIG_DIR}" "outputs"

export DEVICE
export CONFIG_DIR
export GROUP
export EXPERIMENTS
export EVAL_SPLIT
export SAMPLES_PER_CONDITION
export VIS_SEED
export TMP_CONFIG_DIR

echo "Starting ${GROUP} pipeline with nohup..."
echo "Config dir : ${CONFIG_DIR}"
echo "Experiments: ${EXPERIMENTS:-all *.yaml in ${CONFIG_DIR}}"
echo "Device     : ${DEVICE}"
echo "Eval split : ${EVAL_SPLIT}"
echo "Log file   : ${LOG_FILE}"

nohup bash -c '
set -euo pipefail

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "[ERROR] CONFIG_DIR not found: ${CONFIG_DIR}"
  exit 1
fi

CONFIG_FILES=()

if [[ -n "${EXPERIMENTS}" ]]; then
  for EXP in ${EXPERIMENTS}; do
    if [[ "${EXP}" == *.yaml || "${EXP}" == *.yml ]]; then
      CANDIDATE="${CONFIG_DIR}/${EXP}"
    else
      CANDIDATE="${CONFIG_DIR}/${EXP}.yaml"
    fi

    if [[ ! -f "${CANDIDATE}" ]]; then
      echo "[ERROR] Experiment config not found: ${CANDIDATE}"
      exit 1
    fi

    CONFIG_FILES+=("${CANDIDATE}")
  done
else
  while IFS= read -r file; do
    CONFIG_FILES+=("${file}")
  done < <(find "${CONFIG_DIR}" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) | sort)
fi

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No config files found in ${CONFIG_DIR}"
  exit 1
fi

echo "============================================================"
echo "Found ${#CONFIG_FILES[@]} ablation config(s)."
printf " - %s\n" "${CONFIG_FILES[@]}"
echo "============================================================"

for CONFIG_PATH in "${CONFIG_FILES[@]}"; do
  EXPERIMENT_NAME="$(basename "${CONFIG_PATH}")"
  EXPERIMENT_NAME="${EXPERIMENT_NAME%.*}"

  RESULT_DIR="outputs/results/${GROUP}/${EXPERIMENT_NAME}"
  VIS_DIR="outputs/visualizations/${GROUP}/${EXPERIMENT_NAME}"
  ANALYSIS_DIR="outputs/analysis/${GROUP}/${EXPERIMENT_NAME}"
  CHECKPOINT_DIR="outputs/checkpoints/${GROUP}/${EXPERIMENT_NAME}"
  MERGED_CONFIG="${TMP_CONFIG_DIR}/${EXPERIMENT_NAME}.yaml"
  CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_miou.pth"

  mkdir -p "${RESULT_DIR}" "${VIS_DIR}" "${ANALYSIS_DIR}" "${CHECKPOINT_DIR}" "${TMP_CONFIG_DIR}"

  export CONFIG_PATH
  export MERGED_CONFIG
  export GROUP
  export EXPERIMENT_NAME
  export CHECKPOINT_DIR

  python - <<'"'"'PY'"'"'
import os
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] PyYAML is required to create temporary ablation configs.") from exc

config_path = Path(os.environ["CONFIG_PATH"])
merged_config = Path(os.environ["MERGED_CONFIG"])
group = os.environ["GROUP"]
experiment_name = os.environ["EXPERIMENT_NAME"]
checkpoint_dir = os.environ["CHECKPOINT_DIR"]

with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

config.setdefault("experiment", {})
config["experiment"]["group"] = group
config["experiment"]["name"] = experiment_name

config.setdefault("checkpoint", {})
config["checkpoint"]["save_dir"] = checkpoint_dir

if isinstance(config.get("wandb"), dict):
    config["wandb"]["group"] = group
    config["wandb"]["name"] = experiment_name

merged_config.parent.mkdir(parents=True, exist_ok=True)
with merged_config.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PY

  echo ""
  echo "============================================================"
  echo "[${EXPERIMENT_NAME}] Start"
  echo "Original config : ${CONFIG_PATH}"
  echo "Runtime config  : ${MERGED_CONFIG}"
  echo "Checkpoint dir  : ${CHECKPOINT_DIR}"
  echo "Result dir      : ${RESULT_DIR}"
  echo "Visualization   : ${VIS_DIR}"
  echo "Analysis        : ${ANALYSIS_DIR}"
  echo "Device          : ${DEVICE}"
  echo "============================================================"

  echo "[${EXPERIMENT_NAME}] Training..."
  python -m awseg.train \
    --config "${MERGED_CONFIG}" \
    --device "${DEVICE}" \
    --result-dir "${RESULT_DIR}"

  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] Best checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
  fi

  echo "[${EXPERIMENT_NAME}] Evaluation..."
  python -m awseg.evaluate \
    --config "${MERGED_CONFIG}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split "${EVAL_SPLIT}" \
    --device "${DEVICE}" \
    --result-dir "${RESULT_DIR}"

  echo "[${EXPERIMENT_NAME}] Visualization..."
  python -m awseg.visualize \
    --config "${MERGED_CONFIG}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split "${EVAL_SPLIT}" \
    --device "${DEVICE}" \
    --output-dir "${VIS_DIR}" \
    --samples-per-condition "${SAMPLES_PER_CONDITION}" \
    --seed "${VIS_SEED}"

  echo "[${EXPERIMENT_NAME}] Error analysis..."
  python scripts/analyze_errors.py \
    --config "${MERGED_CONFIG}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split "${EVAL_SPLIT}" \
    --condition none \
    --device "${DEVICE}" \
    --output-dir "${ANALYSIS_DIR}"

  echo "[${EXPERIMENT_NAME}] Plot results..."
  python scripts/plot_results.py \
    --result-dir "${RESULT_DIR}" \
    --output-dir "${RESULT_DIR}/plots" || echo "[WARN] plot_results.py failed or does not support these arguments. Skipping plots."

  echo "[${EXPERIMENT_NAME}] Done"
done

echo ""
echo "============================================================"
echo "All ablation experiments finished."
echo "============================================================"
' > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Pipeline started."
echo "PID: ${PID}"
echo "To monitor logs:"
echo "tail -f ${LOG_FILE}"
echo ""
echo "Experiments will run sequentially in the background."