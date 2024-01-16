# Baseline2: LightGCN

## Setup
```bash
cd /opt/ml/input/code/lightgcn
conda init
(base) . ~/.bashrc
(base) conda create -n gcn python=3.10 -y
(base) conda activate gcn
(gcn) pip install -r requirements.txt
(gcn) python train.py
(gcn) python inference.py
```

## Files
`code/lightgcn`
* `train.py`: 학습코드입니다.
* `inference.py`: 추론 후 `submissions.csv` 파일을 만들어주는 소스코드입니다.
* `requirements.txt`: 모델 학습에 필요한 라이브러리들이 정리되어 있습니다.
* `sweep.yaml'`: 하이퍼 파라미터 튜닝을 위한 wandb sweep 설정파일입니다.

`code/lightgcn/lightgcn`
* `args.py`: `argparse`를 통해 학습에 활용되는 여러 argument들을 받아줍니다.
* `datasets.py`: 학습 데이터를 불러 GCN 입력에 맞게 변환해줍니다.
* `trainer.py`: 훈련에 사용되는 함수들을 포함합니다.
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함합니다.

`code/lightgcn/config`
* `default.yaml`: configuration 설정 파일입니다.

## CLI argument override

(gcn) python train.py existing_field=value : 값 변경
(gcn) python train.py +new_field=value : 새 필드 생성 후 값 입력
(gcn) python train.py ++new_or_existing_field=value : 필드가 존재 했다면 값 변경, 필드가 없었다면 생성 후 값 입력

## WandB Sweep

(gcn) wandb sweep sweep.yaml : 이 명렁어는 sweepid를 터미널로 반환합니다.
(gcn) wandb agent sweepid : 반한된 sweepid를 입력해주세요. wandb sweep을 실행합니다.