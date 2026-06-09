#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_nightadapter.sh
#   PHASE=analyze_cls bash scripts/run_nightadapter.sh
#   PHASE=analyze_seg bash scripts/run_nightadapter.sh
#   PHASE=all bash scripts/run_nightadapter.sh
#
# Optional environment variables:
#   DEVICE=cuda:1
#   CONDITION=night
#   CONFIG_PATH=configs/nightadapter.yaml
#   BASELINE_CHECKPOINT_PATH=outputs/checkpoints/proposed/best_miou.pth

CONFIG_PATH="${CONFIG_PATH:-configs/nightadapter.yaml}"
CONDITION="${CONDITION:-night}"
PHASE="${PHASE:-train_eval}"  # analyze_cls, analyze_seg, train_eval, all
DEVICE_ARG=""
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARG="--device ${DEVICE}"
fi

EXPERIMENT_NAME="segformer_b2_nightadapter_stage34"
CHECKPOINT_PATH="outputs/checkpoints/nightadapter/best_miou.pth"
BASELINE_CHECKPOINT_PATH="${BASELINE_CHECKPOINT_PATH:-outputs/checkpoints/proposed/best_miou.pth}"
RESULT_DIR="outputs/results/nightadapter"
VIS_DIR="outputs/visualizations/nightadapter"
ANALYSIS_DIR="outputs/analysis/nightadapter"

mkdir -p "${RESULT_DIR}" "${VIS_DIR}" "${ANALYSIS_DIR}"

if [[ "${PHASE}" == "analyze_cls" || "${PHASE}" == "all" ]]; then
  python -m awseg.analysis.band_rejection \
    --task cls \
    --config "${CONFIG_PATH}" \
    --checkpoint "${BASELINE_CHECKPOINT_PATH}" \
    --condition "${CONDITION}" \
    --stages 2 3 4 \
    --num-bands 8 \
    --output-dir "${ANALYSIS_DIR}" \
    --disable-night-adapter \
    ${DEVICE_ARG}
fi

if [[ "${PHASE}" == "analyze_seg" || "${PHASE}" == "all" ]]; then
  python -m awseg.analysis.band_rejection \
    --task seg \
    --config "${CONFIG_PATH}" \
    --checkpoint "${BASELINE_CHECKPOINT_PATH}" \
    --condition "${CONDITION}" \
    --stages 2 3 4 \
    --num-bands 8 \
    --output-dir "${ANALYSIS_DIR}" \
    --disable-night-adapter \
    ${DEVICE_ARG}
fi

if [[ "${PHASE}" == "train_eval" || "${PHASE}" == "all" ]]; then
  python -m awseg.train \
    --config "${CONFIG_PATH}" \
    --condition "${CONDITION}" \
    --result-dir "${RESULT_DIR}" \
    ${DEVICE_ARG}

  python -m awseg.evaluate \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --split val \
    --condition "${CONDITION}" \
    --result-dir "${RESULT_DIR}" \
    ${DEVICE_ARG}

  if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('awseg.visualize') else 1)
PY
  then
    python -m awseg.visualize \
      --config "${CONFIG_PATH}" \
      --checkpoint "${CHECKPOINT_PATH}" \
      --split val \
      --condition "${CONDITION}" \
      --num-samples 20 \
      --output-dir "${VIS_DIR}" \
      ${DEVICE_ARG}
  else
    echo "[INFO] awseg.visualize module not found. Skipping visualization."
  fi
fi
