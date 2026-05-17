#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_PARENT="${RAW_PARENT:-/home/user/DATA/awseg}"
ENH_PARENT="${ENH_PARENT:-/home/user/DATA/awseg}"
CONDITION="${CONDITION:-night}"

RAW_ROOT="${RAW_PARENT}/data/raw/rgb_anon/${CONDITION}"
ENH_ROOT="${ENH_PARENT}/data/enhanced"

GPU="${GPU:-0}"
SCI_WEIGHT="${SCI_WEIGHT:-medium.pt}"
RETINEX_WEIGHT="${RETINEX_WEIGHT:-LOL_v2_real.pth}"

if [[ -z "${METHOD}" ]]; then
  echo "Usage:"
  echo "  bash scripts/preprocess_deep_enhancement.sh sci"
  echo "  bash scripts/preprocess_deep_enhancement.sh zero_dce"
  echo "  bash scripts/preprocess_deep_enhancement.sh retinexformer"
  echo "  bash scripts/preprocess_deep_enhancement.sh all"
  exit 1
fi

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "[ERROR] RAW_ROOT not found: ${RAW_ROOT}"
  exit 1
fi

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] RAW_ROOT=${RAW_ROOT}"
echo "[INFO] ENH_ROOT=${ENH_ROOT}"
echo "[INFO] CONDITION=${CONDITION}"
echo "[INFO] GPU=${GPU}"

run_sci() {
  echo "[INFO] Running SCI..."

  SCI_ROOT="${PROJECT_ROOT}/third_party/SCI/CVPR"

  find "${RAW_ROOT}" -type d | while read -r INPUT_DIR; do
    NUM_IMAGES="$(find "${INPUT_DIR}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)"

    if [[ "${NUM_IMAGES}" -eq 0 ]]; then
      continue
    fi

    if [[ "${INPUT_DIR}" == "${RAW_ROOT}" ]]; then
      REL_DIR=""
    else
      REL_DIR="${INPUT_DIR#${RAW_ROOT}/}"
    fi

    OUTPUT_DIR="${ENH_ROOT}/sci/rgb_anon/${CONDITION}/${REL_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    echo "[SCI] ${INPUT_DIR} -> ${OUTPUT_DIR}"

    (
      cd "${INPUT_DIR}"

      python "${SCI_ROOT}/test.py" \
        --data_path "." \
        --save_path "${OUTPUT_DIR}" \
        --model "${SCI_ROOT}/weights/${SCI_WEIGHT}" \
        --gpu "${GPU}"
    )
  done

  cd "${PROJECT_ROOT}"
  echo "[DONE] SCI"
}

run_zero_dce() {
  echo "[INFO] Running Zero-DCE..."

  cd "${PROJECT_ROOT}/third_party/Zero-DCE/Zero-DCE_code"

  rm -rf "data/test_data/acdc_${CONDITION}"
  rm -rf "data/result/acdc_${CONDITION}"

  mkdir -p "data/test_data/acdc_${CONDITION}"

  echo "[INFO] Creating image symlinks for Zero-DCE..."

  find "${RAW_ROOT}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | while read -r SRC_PATH; do
    REL_PATH="${SRC_PATH#${RAW_ROOT}/}"
    DST_PATH="data/test_data/acdc_${CONDITION}/${REL_PATH}"

    mkdir -p "$(dirname "${DST_PATH}")"
    ln -s "${SRC_PATH}" "${DST_PATH}"
  done

  python - <<'PY'
from pathlib import Path
import lowlight_test

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
root = Path("data/test_data/acdc_night")

image_paths = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in image_extensions
)

print(f"[INFO] Zero-DCE images: {len(image_paths)}")

for i, path in enumerate(image_paths, start=1):
    lowlight_test.lowlight(str(path))
    if i % 100 == 0:
        print(f"[INFO] processed {i}/{len(image_paths)}")
PY

  mkdir -p "${ENH_ROOT}/zero_dce/rgb_anon/${CONDITION}"

  if [[ ! -d "data/result/acdc_${CONDITION}" ]]; then
    echo "[ERROR] Zero-DCE result directory was not created: data/result/acdc_${CONDITION}"
    exit 1
  fi

  rsync -a "data/result/acdc_${CONDITION}/" "${ENH_ROOT}/zero_dce/rgb_anon/${CONDITION}/"

  cd "${PROJECT_ROOT}"
  echo "[DONE] Zero-DCE"
}

run_retinexformer() {
  echo "[INFO] Running Retinexformer..."

  RETINEX_ROOT="${PROJECT_ROOT}/third_party/Retinexformer"

  cd "${RETINEX_ROOT}"

  mkdir -p tmp_options
  mkdir -p "${ENH_ROOT}/retinexformer/rgb_anon/${CONDITION}"

  if [[ ! -f "pretrained_weights/${RETINEX_WEIGHT}" ]]; then
    echo "[ERROR] Retinexformer weight not found: ${RETINEX_ROOT}/pretrained_weights/${RETINEX_WEIGHT}"
    echo "[INFO] Set RETINEX_WEIGHT or place the weight file under third_party/Retinexformer/pretrained_weights/"
    exit 1
  fi

  find "${RAW_ROOT}" -type d | while read -r INPUT_DIR; do
    NUM_IMAGES="$(
      find "${INPUT_DIR}" -maxdepth 1 -type f \
        \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l
    )"

    if [[ "${NUM_IMAGES}" -eq 0 ]]; then
      continue
    fi

    if [[ "${INPUT_DIR}" == "${RAW_ROOT}" ]]; then
      REL_DIR=""
    else
      REL_DIR="${INPUT_DIR#${RAW_ROOT}/}"
    fi

    OUTPUT_DIR="${ENH_ROOT}/retinexformer/rgb_anon/${CONDITION}/${REL_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    echo "[Retinexformer] ${INPUT_DIR} -> ${OUTPUT_DIR}"

    cat > tmp_options/acdc_${CONDITION}_tmp.yml <<EOF
name: Enhancement_RetinexFormer_ACDC_${CONDITION}
model_type: ImageCleanModel
scale: 1
num_gpu: 1
manual_seed: 100

datasets:
  val:
    name: ValSet
    type: Dataset_PairedImage
    dataroot_gt: ${INPUT_DIR}
    dataroot_lq: ${INPUT_DIR}
    io_backend:
      type: disk

network_g:
  type: RetinexFormer
  in_channels: 3
  out_channels: 3
  n_feat: 40
  stage: 1
  num_blocks: [1,2,2]

path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

train:
  mixing_augs:
    mixup: false
    mixup_beta: 1.2
    use_identity: true

val:
  window_size: 4
  save_img: false
  rgb2bgr: true
  use_image: false
EOF

    PYTHONPATH="${RETINEX_ROOT}:${RETINEX_ROOT}/Enhancement:${PYTHONPATH:-}" \
    python3 Enhancement/test_from_dataset.py \
      --opt tmp_options/acdc_${CONDITION}_tmp.yml \
      --weights "pretrained_weights/${RETINEX_WEIGHT}" \
      --dataset ACDC_${CONDITION} \
      --output_dir "${OUTPUT_DIR}" \
      --gpus "${GPU}"
  done

  cd "${PROJECT_ROOT}"
  echo "[DONE] Retinexformer"
}

case "${METHOD}" in
  sci)
    run_sci
    ;;
  zero_dce)
    run_zero_dce
    ;;
  retinexformer)
    run_retinexformer
    ;;
  all)
    run_sci
    run_zero_dce
    run_retinexformer
    ;;
  *)
    echo "[ERROR] Unknown method: ${METHOD}"
    echo "Available: sci, zero_dce, retinexformer, all"
    exit 1
    ;;
esac

echo "[INFO] Checking output counts..."
echo "Raw:"
find "${RAW_ROOT}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l

if [[ -d "${ENH_ROOT}/sci/rgb_anon/${CONDITION}" ]]; then
  echo "SCI:"
  find "${ENH_ROOT}/sci/rgb_anon/${CONDITION}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l
fi

if [[ -d "${ENH_ROOT}/zero_dce/rgb_anon/${CONDITION}" ]]; then
  echo "Zero-DCE:"
  find "${ENH_ROOT}/zero_dce/rgb_anon/${CONDITION}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l
fi

if [[ -d "${ENH_ROOT}/retinexformer/rgb_anon/${CONDITION}" ]]; then
  echo "Retinexformer:"
  find "${ENH_ROOT}/retinexformer/rgb_anon/${CONDITION}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l
fi