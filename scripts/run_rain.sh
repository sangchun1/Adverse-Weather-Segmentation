#!/usr/bin/env bash
# rain 조건 SegFormer 실험 일괄 실행.
#
# 각 실험: 학습 -> best checkpoint 로 평가. 모두 --condition rain.
# baseline(full FT) 먼저, 그 다음 freeze / dropout / drop_path ablation.
#
# 사용법:
#   bash run_rain.sh              # 전부 실행
#   bash run_rain.sh baseline     # baseline 만
#   bash run_rain.sh freeze       # freeze 실험만
#   bash run_rain.sh dropout      # dropout 실험만
#   bash run_rain.sh droppath     # drop_path 실험만
#
# 프로젝트 루트(configs/, src/ 가 보이는 위치)에서 실행

set -euo pipefail

CONDITION="rain"
PYTHON="${PYTHON:-python}"   

# config exp_id -> config 파일 경로.
# (rain.yaml 이 baseline = full FT = freeze A)
BASELINE_CONFIG="configs/rain/rain.yaml"

FREEZE_CONFIGS=(
  "configs/rain/rain_freeze_B_s1.yaml"
  "configs/rain/rain_freeze_C_s12.yaml"
  "configs/rain/rain_freeze_D_head.yaml"
  "configs/rain/rain_freeze_C_s12_trainnorm.yaml"
)

DROPOUT_CONFIGS=(
  "configs/rain_dropout_0.0.yaml"
  "configs/rain/rain_dropout_0.1.yaml"
  "configs/rain/rain_dropout_0.2.yaml"
)

DROPPATH_CONFIGS=(
  "configs/rain/rain_droppath_0.0.yaml"
  "configs/rain/rain_droppath_0.05.yaml"
  "configs/rain/rain_droppath_0.1.yaml"
  "configs/rain/rain_droppath_0.2.yaml"
)

# 실험 4: encoder/head lr ratio (8 조합)
LR_CONFIGS=(
  "configs/rain/rain_lr_e0.1_h1.0.yaml"
  "configs/rain/rain_lr_e0.1_h3.0.yaml"
  "configs/rain/rain_lr_e0.3_h1.0.yaml"
  "configs/rain/rain_lr_e0.3_h3.0.yaml"
  "configs/rain/rain_lr_e0.5_h1.0.yaml"
  "configs/rain/rain_lr_e0.5_h3.0.yaml"
  "configs/rain/rain_lr_e1.0_h1.0.yaml"
  "configs/rain/rain_lr_e1.0_h3.0.yaml"
)

# config 의 checkpoint.save_dir 를 읽어 best checkpoint 경로를 구한다.
get_ckpt_dir() {
  local config_path="$1"
  "$PYTHON" - "$config_path" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(cfg["checkpoint"]["save_dir"])
PYEOF
}

run_one() {
  local config_path="$1"
  echo ""
  echo "=================================================================="
  echo "[RUN] config: $config_path  (condition=$CONDITION)"
  echo "=================================================================="

  # 1) 학습
  "$PYTHON" -m src.awseg.train --config "$config_path" --condition "$CONDITION"

  # 2) 평가 (best checkpoint)
  local ckpt_dir
  ckpt_dir="$(get_ckpt_dir "$config_path")"
  local best_ckpt="$ckpt_dir/best_miou.pth"

  if [[ -f "$best_ckpt" ]]; then
    echo "[EVAL] $best_ckpt"
    "$PYTHON" -m src.awseg.evaluate --config "$config_path" \
      --checkpoint "$best_ckpt" --split val --condition "$CONDITION"
  else
    echo "[WARN] best checkpoint 없음: $best_ckpt (평가 건너뜀)"
  fi
}

run_group() {
  local -n configs_ref="$1"
  for config_path in "${configs_ref[@]}"; do
    run_one "$config_path"
  done
}

TARGET="${1:-all}"

case "$TARGET" in
  baseline)
    run_one "$BASELINE_CONFIG"
    ;;
  freeze)
    run_group FREEZE_CONFIGS
    ;;
  dropout)
    run_group DROPOUT_CONFIGS
    ;;
  droppath)
    run_group DROPPATH_CONFIGS
    ;;
  lr)
    run_group LR_CONFIGS
    ;;
  all)
    run_one "$BASELINE_CONFIG"      # A (full FT)
    run_group FREEZE_CONFIGS         # B/C/D (+trainnorm)
    run_group DROPOUT_CONFIGS
    run_group DROPPATH_CONFIGS
    run_group LR_CONFIGS             # 실험 4: lr ratio (8 조합)
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Use one of: all | baseline | freeze | dropout | droppath | lr"
    exit 1
    ;;
esac

echo ""
echo "[DONE] target=$TARGET"