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
├── configs/                   # 실행 하이퍼파라미터 설정 디렉토리
│   ├── baseline.yaml               # 공통 baseline 설정
│   ├── fog.yaml                    # fog 실험 설정 (규영)
│   ├── night.yaml                  # night 실험 설정 (상춘)
│   ├── rain.yaml                   # rain 실험 설정 (현민)
│   └── snow.yaml                   # snow 실험 설정 (경민)
│
├── data/                      # 데이터 디렉토리
│   ├── raw/                        # ACDC 원본 데이터 (GitHub에 안 올라감)
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
│   └── splits/                     # 이미지 파일들 경로를 정리해놓은 csv
│       ├── train.csv          
│       ├── val.csv            
│       └── test.csv           
│
├── outputs/                   # 아웃풋 디렉토리 (결과, 체크포인트, 시각화 등)
│   ├── checkpoints/                # 모델 checkpoint (GitHub에 안 올라감)
│   ├── logs/                       # 실행 로그 (GitHub에 안 올라감)
|   ├── results/                    # 실행 결과
│   ├── visualizations/             # 예측 시각화 결과
│   └── wandb/                      # wandb 로그 (GitHub에 안 올라감)
│
├── scripts/                   # 실행 코드 디렉토리
│   ├── prepare_dataset.py          # ACDC 원본 데이터에서 split CSV 생성
│   └── run_baseline.sh             # baseline 학습/평가/시각화 실행
│
└── src/awseg/                 # 핵심 코드 디렉토리
    ├── dataset.py                  # ACDC dataset loader
    ├── train.py                    # 학습 스크립트
    ├── evaluate.py                 # 평가 스크립트
    ├── visualize.py                # 예측 결과 시각화
    ├── metrics.py                  # mIoU, class IoU 계산
    ├── utils.py                    # seed, config, checkpoint 유틸
    ├── logger.py                   # wandb 로그 작성
    ├── models/                     # 모델 코드 (현민)
    ├── losses/                     # loss 코드 (규영)
    └── transforms/                 # transform 및 augmentation 코드 (경민&상춘)
```

## 3. Dataset Preparation

원본 데이터셋은 `data/raw/` 안에 위 구조처럼 보관. 

split CSV 생성을 위해서 다음 명령어를 terminal에 실행:
```bash
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