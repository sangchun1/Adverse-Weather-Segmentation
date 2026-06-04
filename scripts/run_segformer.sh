# 각 실험: 학습 -> best checkpoint 로 평가. 모두 --condition rain.
# baseline(full FT) 먼저, 그 다음 freeze / dropout / drop_path ablation.

# 사용법:
#   bash run_rain.sh              # 전부 실행
#   bash run_rain.sh baseline     # baseline 만
#   bash run_rain.sh freeze       # freeze 실험만
#   bash run_rain.sh dropout      # dropout 실험만
#   bash run_rain.sh droppath     # drop_path 실험만

# 프로젝트 루트(configs/, src/ 가 있는 위치)에서 실행

set -euo pipefail

CONDITION="rain"
PYTHON="${PYTHON:-python}" 

BASELINE_CONFIG="configs/rain/rain.yaml"

FREEZE_CONFIGS=(
  "configs/rain_freeze_B_s1.yaml"
  "configs/rain_freeze_C_s12.yaml"
  "configs/rain_freeze_D_head.yaml"
  "configs/rain_freeze_C_s12_trainnorm.yaml" # stage1,2 freeze +normalization layer는 freeze
)

DROPOUT_CONFIGS=(
  "configs/rain_dropout_0.0.yaml"
  "configs/rain_dropout_0.1.yaml"
  "configs/rain_dropout_0.2.yaml"
)

DROPPATH_CONFIGS=(
  "configs/rain_droppath_0.0.yaml"
  "configs/rain_droppath_0.1.yaml"
  "configs/rain_droppath_0.2.yaml"
)


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
  "$PYTHON" -m awseg.train --config "$config_path" --condition "$CONDITION"

  # 2) 평가 (best checkpoint)
  local ckpt_dir
  ckpt_dir="$(get_ckpt_dir "$config_path")"
  local best_ckpt="$ckpt_dir/best_miou.pth"

  if [[ -f "$best_ckpt" ]]; then
    echo "[EVAL] $best_ckpt"
    "$PYTHON" -m awseg.evaluate --config "$config_path" \
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
  all)
    run_one "$BASELINE_CONFIG"      # A (full FT)
    run_group FREEZE_CONFIGS         # B/C/D (+trainnorm)
    run_group DROPOUT_CONFIGS
    run_group DROPPATH_CONFIGS
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Use one of: all | baseline | freeze | dropout | droppath"
    exit 1
    ;;
esac

echo ""
echo "[DONE] target=$TARGET"