# import os
from pathlib import Path

BASE_DIR = Path("D:/KCH/workspace/Myungsung")
# BASE_DIR = Path("D:/AISystem")

# 학습 데이터셋 저장 경로
DATASETS_DIR = BASE_DIR / "datasets"

TRAIN_DIR = BASE_DIR / "datasets" / "train"
VAL_DIR = BASE_DIR / "datasets" / "val"
TEST_DIR = BASE_DIR / "datasets" / "test"

# CLASSES 목록 데이터 경로
CLASSES_TXT = DATASETS_DIR / "classes.txt"

LABELED_TRAIN_DIR = BASE_DIR / "datasets" / "labeled" / "train"
LABELED_VAL_DIR = BASE_DIR / "datasets" / "labeled" / "val"
LABELED_TEST_DIR = BASE_DIR / "datasets" / "labeled" / "test"

# 실시간 추론용
INFER_INP_IMG_DIR = DATASETS_DIR / "inference" / "input"
INFER_OUT_IMG_DIR = DATASETS_DIR / "inference" / "output"

"""
datasets/
│
├─ classes.txt
├─ final_datsets.txt
│
├─ train/
│   ├─ images/
│   │   ├─ Line03/ # 일자별로 디렉토리 존재
│   │   └─ Line04/ # 일자별로 디렉토리 존재
│   └─ labels/
│       ├─ Line03/ # 일자별로 디렉토리 존재
│       └─ Line04/ # 일자별로 디렉토리 존재
│
├─ val/
│   ├─ images/ ...
│   └─ labels/ ...
│
├─ test/
│   ├─ images/
│   └─ labels/
│
├─ labeled/
│   ├─ train/
│   ├─ val/
│   └─ test/
│
└─ inference/
    ├─ input/
    └─ output/
"""


# 모델 경로
MODELS_DIR = BASE_DIR / "models"
"""
models/
│
├─ train_YYYYMMDDHHMMSS/
│   └─ weights/
│
├─ train_YYYYMMDDHHMMSS/
│   └─ weights/
│ ... 
"""


# YOLO 
RUNS_DIR = BASE_DIR / "runs"
DETECT_DIR = RUNS_DIR / "detect"
PREDICT_DIR = DETECT_DIR / "predict"

"""
runs/
│
├─ detect/
│   ├─ train/
│   └─ val/
│
├─ predict/
│   └─ weights/
│ ... 
"""

# Shared Folder
SHARED = Path("D:/image/workspace/Myungsung/shared")
