from pathlib import Path
# =================
# Tables
# =================
PROD_CNT_TABLE = ""
TRAIN_INFO_TABLE = ""
ORG_IMG_TABLE = ""
MONITOR_IMG_TABLE = ""

# =================
# Paths
# =================
BASE_DIR = ""

# Directory to store dataset files
DATASETS_DIR = BASE_DIR / ""
TRAIN_DIR = BASE_DIR / "datasets" / "train"
VAL_DIR = BASE_DIR / "datasets" / "val"
TEST_DIR = BASE_DIR / "datasets" / "test"

# Class name text file
CLASSES_TXT = DATASETS_DIR / "classes.txt"

# Labeled Dataset Directory
LABELED_TRAIN_DIR = BASE_DIR / "datasets" / "labeled" / "train"
LABELED_VAL_DIR = BASE_DIR / "datasets" / "labeled" / "val"
LABELED_TEST_DIR = BASE_DIR / "datasets" / "labeled" / "test"

# Inference Directory
INFER_INP_IMG_DIR = DATASETS_DIR / "inference" / "input"
INFER_OUT_IMG_DIR = DATASETS_DIR / "inference" / "output"

# Model Directory
MODELS_DIR = BASE_DIR / "models"

# YOLO 
RUNS_DIR = BASE_DIR / "runs"
DETECT_DIR = RUNS_DIR / "detect"
PREDICT_DIR = DETECT_DIR / "predict"

# Shared Folder
SHARED = ""
