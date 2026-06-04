#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash scripts/run_enhancement.sh
#   bash scripts/run_enhancement.sh configs/proposed.yaml
#   bash scripts/run_enhancement.sh configs/proposed.yaml fog
#   CONDITION=fog bash scripts/run_enhancement.sh
#   CONFIG_DIR=configs/enhancement bash scripts/run_enhancement.sh
#   EXPERIMENTS="none gamma clahe gamma_clahe" bash scripts/run_enhancement.sh
#
# This script runs enhancement experiments sequentially.
# Each experiment config in configs/enhancement/ is treated as an override config.
# The override config is recursively merged into BASE_CONFIG, then executed.
#
# If CONDITION is empty, all weather conditions are used.
# If CONDITION is set to fog/rain/snow/night, train/evaluate/visualize/analyze use only that condition.
#
# Output structure:
#   outputs/logs/enhancement_YYYYMMDD_HHMMSS.log
#   outputs/results/enhancement/<experiment>/
#   outputs/visualizations/enhancement/<experiment>/
#   outputs/analysis/enhancement/<experiment>/
#   outputs/checkpoints/enhancement/<experiment>/

BASE_CONFIG="${1:-configs/proposed.yaml}"
CONDITION="${2:-${CONDITION:-}}"
DEVICE="${3:-${DEVICE:-cuda:0}}"

GROUP="enhancement"
CONFIG_DIR="${CONFIG_DIR:-configs/enhancement}"
EXPERIMENTS="${EXPERIMENTS:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -n "${CONDITION}" ]]; then
  LOG_RUN_NAME="${GROUP}_${CONDITION}_${TIMESTAMP}"
else
  LOG_RUN_NAME="${GROUP}_${TIMESTAMP}"
fi

LOG_DIR="outputs/logs"
LOG_FILE="${LOG_DIR}/${LOG_RUN_NAME}.log"
TMP_CONFIG_DIR="outputs/tmp_configs/${GROUP}"

EVAL_SPLIT="${EVAL_SPLIT:-val}"
SAMPLES_PER_CONDITION="${SAMPLES_PER_CONDITION:-5}"
VIS_SEED="${VIS_SEED:-42}"

mkdir -p "${LOG_DIR}" "${TMP_CONFIG_DIR}" "outputs"

export BASE_CONFIG
export CONDITION
export DEVICE
export GROUP
export CONFIG_DIR
export EXPERIMENTS
export TMP_CONFIG_DIR
export EVAL_SPLIT
export SAMPLES_PER_CONDITION
export VIS_SEED

echo "Starting ${GROUP} experiment pipeline with nohup..."
echo "Base config : ${BASE_CONFIG}"
echo "Config dir  : ${CONFIG_DIR}"
echo "Experiments : ${EXPERIMENTS:-all *.yaml in ${CONFIG_DIR}}"

if [[ -n "${CONDITION}" ]]; then
  echo "Condition   : ${CONDITION}"
else
  echo "Condition   : all"
fi

echo "Log file    : ${LOG_FILE}"
echo "Device      : ${DEVICE}"
echo "Eval split  : ${EVAL_SPLIT}"

nohup bash -c '
set -euo pipefail

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERROR] BASE_CONFIG not found: ${BASE_CONFIG}"
  exit 1
fi

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
echo "Found ${#CONFIG_FILES[@]} ${GROUP} experiment config(s)."
printf " - %s\n" "${CONFIG_FILES[@]}"
echo "============================================================"

for OVERRIDE_CONFIG in "${CONFIG_FILES[@]}"; do
  EXP_STEM="$(basename "${OVERRIDE_CONFIG}")"
  EXP_STEM="${EXP_STEM%.*}"

  if [[ -n "${CONDITION}" ]]; then
    RUN_NAME="${EXP_STEM}_${CONDITION}"
  else
    RUN_NAME="${EXP_STEM}"
  fi

  RESULT_DIR="outputs/results/${GROUP}/${RUN_NAME}"
  VIS_DIR="outputs/visualizations/${GROUP}/${RUN_NAME}"
  ANALYSIS_DIR="outputs/analysis/${GROUP}/${RUN_NAME}"
  CHECKPOINT_DIR="outputs/checkpoints/${GROUP}/${RUN_NAME}"
  MERGED_CONFIG="${TMP_CONFIG_DIR}/${RUN_NAME}.yaml"
  CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_miou.pth"

  mkdir -p "${RESULT_DIR}" "${VIS_DIR}" "${ANALYSIS_DIR}" "${CHECKPOINT_DIR}" "${TMP_CONFIG_DIR}"

  export OVERRIDE_CONFIG
  export RUN_NAME
  export RESULT_DIR
  export VIS_DIR
  export ANALYSIS_DIR
  export CHECKPOINT_DIR
  export MERGED_CONFIG

  python - <<PY
import os
from pathlib import Path

import yaml

base_path = Path(os.environ["BASE_CONFIG"])
override_path = Path(os.environ["OVERRIDE_CONFIG"])
merged_path = Path(os.environ["MERGED_CONFIG"])

group = os.environ["GROUP"]
run_name = os.environ["RUN_NAME"]
result_dir = os.environ["RESULT_DIR"]
vis_dir = os.environ["VIS_DIR"]
analysis_dir = os.environ["ANALYSIS_DIR"]
checkpoint_dir = os.environ["CHECKPOINT_DIR"]

with base_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

with override_path.open("r", encoding="utf-8") as f:
    override = yaml.safe_load(f) or {}

def recursive_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            recursive_update(dst[key], value)
        else:
            dst[key] = value
    return dst

recursive_update(cfg, override)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = run_name
cfg["experiment"]["group"] = group

cfg.setdefault("checkpoint", {})
cfg["checkpoint"]["save_dir"] = checkpoint_dir
cfg["checkpoint"].setdefault("save_best_name", "best_miou.pth")
cfg["checkpoint"].setdefault("save_last_name", "last.pth")

cfg.setdefault("evaluate", {})
cfg["evaluate"]["output_dir"] = result_dir

cfg.setdefault("output", {})
cfg["output"]["checkpoint_dir"] = checkpoint_dir
cfg["output"]["result_dir"] = result_dir
cfg["output"]["log_dir"] = f"outputs/logs/{group}"
cfg["output"]["visualization_dir"] = vis_dir
cfg["output"]["analysis_dir"] = analysis_dir

cfg.setdefault("wandb", {})
if cfg["wandb"].get("enabled", False):
    cfg["wandb"]["run_name"] = f"{group}_{run_name}"
    tags = cfg["wandb"].get("tags", [])
    if tags is None:
        tags = []
    tags = list(tags)
    for tag in [group, run_name]:
        if tag not in tags:
            tags.append(tag)
    cfg["wandb"]["tags"] = tags

merged_path.parent.mkdir(parents=True, exist_ok=True)
with merged_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"[OK] Wrote merged config: {merged_path}")
PY

  echo ""
  echo "============================================================"
  echo "Running ${GROUP} experiment: ${RUN_NAME}"
  echo "Override config : ${OVERRIDE_CONFIG}"
  echo "Merged config   : ${MERGED_CONFIG}"
  echo "Checkpoint path : ${CHECKPOINT_PATH}"
  echo "Result dir      : ${RESULT_DIR}"
  echo "Visualization   : ${VIS_DIR}"
  echo "Analysis        : ${ANALYSIS_DIR}"
  echo "============================================================"

  echo "[1/5] Training"
  python -m awseg.train \
    --config "${MERGED_CONFIG}" \
    --result-dir "${RESULT_DIR}" \
    "${DEVICE_ARGS[@]}" \
    "${CONDITION_ARGS[@]}"

  echo ""
  echo "[2/5] Evaluating"
  python -m awseg.evaluate \
    --config "${MERGED_CONFIG}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split "${EVAL_SPLIT}" \
    --result-dir "${RESULT_DIR}" \
    "${DEVICE_ARGS[@]}" \
    "${CONDITION_ARGS[@]}"

  echo ""
  echo "[3/5] Visualizing"
  python -m awseg.visualize \
    --config "${MERGED_CONFIG}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split "${EVAL_SPLIT}" \
    --output-dir "${VIS_DIR}" \
    "${DEVICE_ARGS[@]}" \
    --samples-per-condition "${SAMPLES_PER_CONDITION}" \
    --shuffle \
    --seed "${VIS_SEED}" \
    "${CONDITION_ARGS[@]}"

  echo ""
  echo "[4/5] Analyzing errors"
  python scripts/analyze_errors.py \
    --group "${GROUP}" \
    --experiment "${RUN_NAME}" \
    --condition "${ANALYZE_CONDITION}" \
    --config "${MERGED_CONFIG}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output-dir "${ANALYSIS_DIR}" \
    "${DEVICE_ARGS[@]}"

  echo ""
  echo "[DONE] ${GROUP} experiment finished: ${RUN_NAME}"
done

echo ""
echo "============================================================"
echo "[5/5] Plotting ${GROUP} results"
echo "============================================================"

python scripts/plot_results.py \
  --group "${GROUP}" \
  --results-root "outputs/results" \
  --output-dir "outputs/visualizations/${GROUP}/plots"

echo ""
echo "============================================================"
echo "${GROUP} experiment pipeline finished successfully."
echo "Log file: ${LOG_FILE}"
echo "============================================================"
' > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Pipeline started."
echo "PID: ${PID}"
echo "To monitor logs:"
echo "tail -f ${LOG_FILE}"
echo ""
echo "Experiments will run sequentially in the background."
