# 악천후 주행 환경에서의 강건한 Semantic Segmentation

<p align="center">
  <b>Adverse-Weather-Segmentation</b><br>
  ACDC 기반 악천후 주행 장면에서 semantic segmentation 모델의 강건성을 분석하고 개선하는 프로젝트
</p>

<p align="center">
  <a href="https://github.com/sangchun1/Adverse-Weather-Segmentation">Repository</a>
  · <a href="#설치">Installation</a>
  · <a href="#재현-파이프라인">Reproduction</a>
  · <a href="#실험-결과">Results</a>
</p>

---

## 개요

악천후 주행 환경에서는 비, 안개, 눈, 야간 조명 변화로 인해 영상의 밝기, 대비, 가시성, texture 정보가 크게 달라진다. 이러한 변화는 자율주행 perception 시스템의 핵심 task인 semantic segmentation 성능을 저하시킬 수 있다.

본 프로젝트는 ACDC 기반 악천후 주행 데이터를 사용하여 U-Net baseline과 SegFormer 기반 모델을 비교하고, loss function, weather-specific augmentation, image enhancement가 악천후 조건별 segmentation 성능에 어떤 영향을 주는지 분석한다.

이 프로젝트는 다음 질문을 중심으로 진행된다.

- 악천후 조건에서는 normal 조건 대비 segmentation 성능이 어떻게 달라지는가?
- U-Net과 SegFormer는 rain, fog, snow, night 조건에서 어떤 차이를 보이는가?
- Loss function, augmentation, enhancement 중 어떤 접근이 조건별 강건성 개선에 효과적인가?
- 전체 mIoU 개선과 특정 weather condition 개선 사이에 trade-off가 존재하는가?

<p align="center">
  <img src="assets/figures/overview_pipeline.png" width="850" alt="Overall pipeline">
</p>
<p align="center">
  <b>Figure 1.</b> 전체 연구 파이프라인. <i>TODO: 최종 pipeline figure 추가</i>
</p>

---

## 주요 방법

본 프로젝트는 먼저 CNN 기반 U-Net을 초기 baseline으로 사용하고, 이후 SegFormer를 주요 비교 기준으로 설정한다. 이후 모든 개선 실험은 SegFormer baseline을 기준으로 수행한다.

### 1. Baseline Models

- **U-Net**  
  Encoder-decoder 구조와 skip connection을 사용하는 CNN 기반 semantic segmentation baseline.

- **SegFormer**  
  Hierarchical transformer encoder와 lightweight decoder를 사용하는 transformer 기반 segmentation 모델. 본 프로젝트의 주요 실험 backbone으로 사용한다.

### 2. Loss Function

Class imbalance와 hard pixel 문제를 완화하기 위해 여러 segmentation loss를 비교한다.

| Loss | 목적 | 비고 |
|---|---|---|
| Cross-Entropy | 기본 픽셀 단위 분류 손실 | Baseline |
| Dice Loss | 작은 객체 및 class imbalance 완화 | TODO |
| Focal Loss | hard pixel에 더 큰 가중치 부여 | TODO |
| Tversky Loss | false positive / false negative 균형 조절 | TODO |
| Hybrid Loss | CE, Dice, Focal, Tversky 계열 조합 | TODO |

### 3. Weather-specific Augmentation

일반적인 color jitter뿐 아니라 악천후 조건의 시각적 degradation을 반영한 augmentation을 적용한다.

| Augmentation | 적용 목적 | 대상 조건 |
|---|---|---|
| Flip / Crop | 기본적인 공간적 일반화 | All |
| Color Jitter | 밝기, 대비, 채도 변화에 대한 강건성 | All |
| Synthetic Rain | 빗줄기 및 contrast 변화 반영 | Rain |
| Synthetic Fog | haze 및 저가시성 반영 | Fog |
| Synthetic Snow | snow particle 및 밝기 변화 반영 | Snow |
| Synthetic Night | 저조도 및 조명 변화 반영 | Night |

<p align="center">
  <img src="assets/figures/augmentation_examples.png" width="850" alt="Augmentation examples">
</p>
<p align="center">
  <b>Figure 2.</b> Augmentation 적용 예시. <i>TODO: original / color jitter / weather-specific augmentation 비교 이미지 추가</i>
</p>

### 4. Image Enhancement

입력 영상의 밝기와 대비를 보정하여 low-visibility condition에서 segmentation 성능이 개선되는지 분석한다.

| Enhancement | 설명 | 기대 효과 |
|---|---|---|
| Gamma Correction | 전체 밝기 분포 보정 | Night 조건의 저조도 영역 개선 |
| CLAHE | 지역 대비 향상 | Fog / Night 조건의 객체 경계 강조 |
| Gamma + CLAHE | 밝기 보정과 지역 대비 향상 결합 | 단일 enhancement 대비 보완 효과 분석 |

<p align="center">
  <img src="assets/figures/enhancement_examples.png" width="850" alt="Enhancement examples">
</p>
<p align="center">
  <b>Figure 3.</b> Image enhancement 적용 예시. <i>TODO: original / gamma / CLAHE / gamma+CLAHE 비교 이미지 추가</i>
</p>

---

## 데이터셋

### ACDC 기반 악천후 주행 데이터

본 프로젝트는 ACDC 기반의 악천후 주행 장면 데이터를 사용한다. Semantic segmentation label은 Cityscapes-style 19개 class를 기준으로 한다.

- **Task**: 19-class semantic segmentation
- **Conditions**: normal, rain, fog, snow, night
- **Input**: RGB driving scene image
- **Label**: pixel-wise semantic mask
- **Main metric**: mIoU, condition-wise mIoU, class-wise IoU

### Class 목록

```text
road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
vegetation, terrain, sky, person, rider, car, truck, bus, train,
motorcycle, bicycle
```

### 데이터 구조

아래와 같은 구조로 `data/raw/`에 배치한 뒤 split CSV를 생성한다.

```text
data/
├── raw/
│   ├── rgb_anon/
│   │   ├── fog/
│   │   ├── night/
│   │   ├── rain/
│   │   └── snow/
│   └── gt/
│       ├── fog/
│       ├── night/
│       ├── rain/
│       └── snow/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

> `data/raw/`에는 ACDC 원본 이미지와 label을 직접 배치해야 한다. 데이터셋 라이선스 문제로 원본 이미지는 이 repository에 포함하지 않는다.

---

## 평가

모델 성능은 전체 성능과 조건별 성능을 함께 평가한다.

- **Overall mIoU**: 전체 validation/test set 기준 평균 IoU
- **Condition-wise mIoU**: normal, rain, fog, snow, night 조건별 mIoU
- **Class-wise IoU**: 19개 semantic class별 IoU
- **Qualitative result**: RGB image, ground truth, prediction 비교
- **Error map**: 오분류 픽셀 위치 시각화

<p align="center">
  <img src="assets/figures/qualitative_examples.png" width="850" alt="Qualitative examples">
</p>
<p align="center">
  <b>Figure 4.</b> 정성적 결과 예시. <i>TODO: RGB / GT / baseline prediction / proposed prediction / error map 추가</i>
</p>

---

## 실험 결과

아래 표의 값은 최종 실험 결과가 정리되는 대로 채운다.

### 1. U-Net vs SegFormer Baseline

| Model | Overall mIoU | Normal | Rain | Fog | Snow | Night |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | TODO | TODO | TODO | TODO | TODO | TODO |
| SegFormer | TODO | TODO | TODO | TODO | TODO | TODO |

<p align="center">
  <img src="assets/figures/baseline_condition_miou.png" width="750" alt="Baseline condition-wise mIoU">
</p>
<p align="center">
  <b>Figure 5.</b> U-Net과 SegFormer의 condition-wise mIoU 비교. <i>TODO</i>
</p>

### 2. Loss Function 비교

| Model | Loss | Overall mIoU | Rain | Fog | Snow | Night |
|---|---|---:|---:|---:|---:|---:|
| SegFormer | Cross-Entropy | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Dice | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Focal | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Tversky | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Hybrid | TODO | TODO | TODO | TODO | TODO |

### 3. Augmentation 비교

| Model | Augmentation | Overall mIoU | Rain | Fog | Snow | Night |
|---|---|---:|---:|---:|---:|---:|
| SegFormer | None | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Color Jitter | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Weather-specific | TODO | TODO | TODO | TODO | TODO |
| SegFormer | All | TODO | TODO | TODO | TODO | TODO |

### 4. Image Enhancement 비교

| Model | Enhancement | Overall mIoU | Rain | Fog | Snow | Night |
|---|---|---:|---:|---:|---:|---:|
| SegFormer | None | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Gamma | TODO | TODO | TODO | TODO | TODO |
| SegFormer | CLAHE | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Gamma + CLAHE | TODO | TODO | TODO | TODO | TODO |

### 5. 최종 모델 비교

| Model | Setting | Overall mIoU | Normal | Rain | Fog | Snow | Night |
|---|---|---:|---:|---:|---:|---:|---:|
| U-Net | Baseline | TODO | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Baseline | TODO | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Best Loss | TODO | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Best Augmentation | TODO | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Best Enhancement | TODO | TODO | TODO | TODO | TODO | TODO |
| SegFormer | Proposed | TODO | TODO | TODO | TODO | TODO | TODO |

### 주요 분석 요약

- **TODO**: 전체 mIoU 기준 가장 좋은 방법 정리
- **TODO**: rain / fog / snow / night 조건별 가장 효과적인 방법 정리
- **TODO**: 특정 condition 개선이 전체 성능에 미친 trade-off 정리
- **TODO**: class-wise IoU에서 개선 또는 악화가 뚜렷한 class 정리

---

## 설치

### 1. Repository clone

```bash
git clone https://github.com/sangchun1/Adverse-Weather-Segmentation.git
cd Adverse-Weather-Segmentation
```

### 2. Conda 환경 생성

```bash
conda create -n awseg python=3.10 -y
conda activate awseg
```

### 3. PyTorch 설치

사용 중인 CUDA 버전에 맞는 PyTorch를 먼저 설치한다.

예시: CUDA 12.8

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

CUDA 버전이 다르면 PyTorch 공식 설치 가이드에 맞게 명령어를 수정한다.

### 4. Package 설치

```bash
pip install -U pip
pip install -e .
```

개발용 dependency까지 설치하려면 다음을 사용한다.

```bash
pip install -e ".[dev]"
```

---

## 재현 파이프라인

일반적인 전체 실행 흐름은 다음과 같다.

```bash
# 1. 환경 활성화
conda activate awseg

# 2. 데이터 split CSV 생성
python scripts/prepare_dataset.py

# 3. Baseline 학습
bash scripts/run_baseline.sh

# 4. 모델 비교 실험
bash scripts/run_model.sh

# 5. Loss 실험
# 예: configs/loss/*.yaml 설정을 사용하여 실행
# bash scripts/<loss_run_script>.sh

# 6. Augmentation 실험
# 예: configs/augmentation/*.yaml 설정을 사용하여 실행
# bash scripts/<augmentation_run_script>.sh

# 7. Enhancement 실험
bash scripts/run_enhancement.sh

# 8. 최종 proposed 설정 학습
bash scripts/run_proposed.sh
```

---

## 데이터셋 준비

`data/raw/` 아래에 RGB image와 ground-truth label을 배치한 뒤 split CSV를 생성한다.

```bash
python scripts/prepare_dataset.py
```

생성 결과는 다음과 같다.

```text
data/splits/
├── train.csv
├── val.csv
└── test.csv
```

각 CSV는 학습, 검증, 테스트에 사용할 image path, label path, condition 정보를 포함한다.

---

## 학습

### Baseline 학습

```bash
bash scripts/run_baseline.sh
```

### SegFormer / 모델 비교 실험

```bash
bash scripts/run_model.sh
```

### Enhancement 실험

```bash
bash scripts/run_enhancement.sh
```

### Proposed 모델 학습

```bash
bash scripts/run_proposed.sh
```

학습 결과는 기본적으로 `outputs/` 아래에 저장된다.

```text
outputs/
├── checkpoints/
├── logs/
├── results/
├── visualizations/
├── analysis/
└── wandb/
```

---

## 평가

학습된 checkpoint를 사용하여 validation 또는 test split에서 평가한다.

```bash
python -m awseg.evaluate \
  --config configs/proposed.yaml \
  --checkpoint outputs/checkpoints/proposed/best_miou.pth \
  --split val \
  --device cuda:0
```

다른 GPU를 사용하려면 `--device cuda:1`처럼 수정한다.

---

## 시각화 및 Error Analysis

### 결과 plot 생성

```bash
python scripts/plot_results.py
```

### Error map 및 대표 실패/성공 사례 분석

```bash
python scripts/analyze_errors.py \
  --condition night \
  --device cuda:0
```

`condition`은 다음 중 하나를 사용할 수 있다.

```text
none, rain, fog, snow, night
```

- `none`: 전체 validation set 기준 분석
- `rain`, `fog`, `snow`, `night`: 해당 악천후 조건만 분석

---

## 향후 정리할 항목

- 최종 method별 mIoU table 업데이트
- condition-wise mIoU plot 추가
- class-wise IoU plot 추가
- representative qualitative result 추가
- error map figure 추가
- 최종 best config와 checkpoint 경로 정리
- 논문 형식의 결과 해석 및 한계점 정리

---

## Citation

이 repository를 사용하거나 참고하는 경우 아래 형식으로 인용할 수 있다.

```bibtex
@misc{adverseweathersegmentation2026,
  title  = {Robust Semantic Segmentation under Adverse Weather Conditions},
  author = {Adverse-Weather-Segmentation Team},
  year   = {2026},
  url    = {https://github.com/sangchun1/Adverse-Weather-Segmentation}
}
```

---

## Acknowledgements

본 프로젝트는 이미지 데이터 분석을 위한 딥러닝 강의의 팀 프로젝트로 수행되었다. 악천후 주행 장면에서 semantic segmentation 모델의 강건성을 분석하고, 조건별 성능 개선 방법을 비교하는 것을 목표로 한다.
