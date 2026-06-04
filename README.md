# Adverse-Weather-Segmentation
악천후 주행 환경에서 semantic segmentation 모델의 강건성 분석

## 1. 프로젝트 개요
- 데이터셋: ACDC
- 태스크: 19-클래스 semantic segmentation
- 날씨조건: fog, night, rain, snow
- 베이스라인 모델: U-Net from scratch
- 메인 평가 지표: mIoU, class-wise IoU, condition-wise mIoU

## 2. 디렉토리(폴더) 구조

```text
Adverse-Weather-Segmentation/
├── configs/
│   ├── baseline.yaml
│   ├── proposed.yaml
│   │
│   ├── loss/
│   │   ├── ce.yaml
│   │   ├── dice.yaml
│   │   ├── focal.yaml
│   │   ├── tversky.yaml
│   │   ├── focal_tversky.yaml
│   │   ├── ce_dice.yaml
│   │   ├── ce_tversky.yaml
│   │   ├── ce_lovasz.yaml
│   │   ├── ohem_tversky.yaml
│   │   ├── tversky_lovasz.yaml
│   │   ├── ce_focal_dice.yaml
│   │   └── ce_focal_tversky.yaml
│   │
│   ├── model/
│   │   ├── unet.yaml
│   │   └── segformer.yaml
│   │
│   ├── augmentation/
│   │   ├── flip.yaml
│   │   ├── jitter.yaml
│   │   ├── class_crop.yaml
│   │   ├── weather_fog.yaml
│   │   ├── weather_rain.yaml
│   │   ├── weather_snow.yaml
│   │   ├── weather_night.yaml
│   │   ├── weather_mixed.yaml
│   │   └── all.yaml
│   │
│   └── enhancement/
│       ├── gamma.yaml
│       ├── clahe.yaml
│       └── gamma_clahe.yaml
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── colab_run_loss.ipynb
│   ├── colab_run_augmentation.ipynb
│   ├── run_model.sh
│   ├── run_enhancement.sh
│   ├── run_proposed.sh
│   ├── colab_run_proposed.ipynb
│   ├── run_baseline.sh
│   ├── colab_run_baseline.ipynb
│   ├── analyze_errors.py
│   ├── colab_analyze_errors.ipynb
│   ├── plot_results.py
│   ├── colab_plot_results.ipynb
│   └── tune_segformer.py
│
├── src/
│   └── awseg/
│       ├── __init__.py
│       ├── dataset.py
│       ├── train.py
│       ├── evaluate.py
│       ├── metrics.py
│       ├── utils.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── builder.py
│       │   ├── unet.py
│       │   └── segformer.py
│       │
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── builder.py
│       │   ├── cross_entropy.py
│       │   ├── dice.py
│       │   ├── focal.py
│       │   ├── tversky.py
│       │   ├── lovasz.py
│       │   ├── ohem.py
│       │   └── hybrid.py
│       │
│       └── transforms/
│           ├── __init__.py
│           ├── transform.py
│           ├── enhancement.py
│           ├── augmentation.py
│           └── weather_augmentation.py
│
├── outputs/
|   ├── results/
|   |   ├── baseline/
|   |   ├── loss/
|   |   ├── model/
|   |   ├── augmentation/
|   |   ├── enhancement/
|   |   └── proposed/
│   ├── analysis/
│   ├── checkpoints/                
│   ├── logs/                                       
│   ├── visualizations/             
│   └── wandb/                      
│
├── data/
│   ├── raw/                       
│   |   ├── rgb_anon/
|   |   |   ├── fog/
|   |   |   |   ├── train/
|   |   |   |   ├── train_ref/
|   |   |   |   ├── val/
|   |   |   |   ├── val_ref/
|   |   |   |   ├── test/
|   |   |   |   └── test_ref/
|   |   |   ├── night/
|   |   |   ├── rain/
|   |   |   └── snow/
|   |   |
|   |   └── gt/
|   |       ├── fog/
|   |       |   ├── train/
|   |       |   └── val/
|   |       ├── night/
|   |       ├── rain/
|   |       └── snow/
|   |
│   └── splits/                     
│       ├── train.csv          
│       ├── val.csv            
│       └── test.csv 
│
├── pyproject.toml
├── README.md
└── .gitignore
```

## 3. 데이터셋 준비

원본 데이터셋은 `data/raw/` 안에 위 구조처럼 보관. 

split CSV 생성을 위해서 다음 명령어를 terminal에 실행:
```bash
conda activate awseg
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 # 본인 cuda 버전에 맞게 수정
pip install -e .
python scripts/prepare_dataset.py
```

## 4. 수정하면 안되는 파일
다음 파일은 공통 파이프라인이므로 수정 전 팀원과 상의해주세요!

```
src/awseg/dataset.py
src/awseg/metrics.py
src/awseg/train.py
src/awseg/evaluate.py
src/awseg/utils.py
src/awseg/logger.py
scripts/prepare_dataset.py
configs/baseline.yaml
```

## 5. 코드 추가하는 위치

### [현민] 모델 코드 추가
```
src/awseg/models/<model_name>.py
```
그리고:
```
src/awseg/models/builder.py
```
에 등록

### [규영] Loss 코드 추가
```
src/awseg/losses/<loss_name>.py
```
그리고:
```
src/awseg/losses/builder.py
```
에 드록

### [경민] Augmentation 코드 추가
```
src/awseg/transforms/basic_aug.py
src/awseg/transforms/weather_aug.py
```
권장 분류:
```
basic_aug.py   → flip, crop, blur, color jitter
weather_aug.py → synthetic fog/rain/snow/night augmentation
```

## 6. 실행 방법

코드 추가 방법:
1. 본인 실험 설정을 configs/에 본인 yaml 파일에 추가
2. models/losses/transforms 중 담당 영역에 새 파일을 추가
3. builder.py에 등록
4. run_baseline.sh를 참고하여 실행 sh 파일 작성

코드 실행 방법:
1. `wandb login`
2. `conda activate awseg`
3. `bash scripts/<실행파일이름>.sh`

깃허브에 올리는 방법:
1. `git switch <본인 브랜치 이름>`
2. `git add .` -> 특정 파일만 추가하고 싶으면 "." 대신 파일경로 추가 (예: `git add scripts/run.sh`)
3. `git commit -m "<커밋 이름>"`
4. `git pull origin <본인 branch 이름>`
5. `git push origin <본인 branch 이름>`

브랜치 배정
- 현민: `fog`
- 규영: `rain`
- 경민: `snow`
- 상춘: `night`
