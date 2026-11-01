"""Central configuration for the hand gesture recognition project."""
import os
from pathlib import Path

import certifi
import torch

# python.org builds on macOS ship without CA certs; needed to fetch pretrained weights
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_FILE = DATA_DIR / "leapgestrecog_128.npz"
METADATA_FILE = DATA_DIR / "metadata.csv"
REMOVED_FILE = DATA_DIR / "removed_files.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CKPT_DIR = ROOT / "checkpoints"

IMG_SIZE = 128
NUM_CLASSES = 10
SEED = 42

# Gesture folder names in LeapGestRecog, in label order 0..9
GESTURES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down",
]
CLASS_NAMES = [g.split("_", 1)[1] for g in GESTURES]

# Subject-wise split (Section 2.5 of the report): 7 train / 1 val / 2 test
TRAIN_SUBJECTS = [0, 1, 2, 3, 4, 5, 6]
VAL_SUBJECTS = [7]
TEST_SUBJECTS = [8, 9]

# ImageNet statistics used by the pretrained MobileNetV2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


for d in (DATA_DIR, RESULTS_DIR, FIG_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)
