# 1/8. Configuration: imports, seed, dataset and model paths, hyperparameters, checkpoint paths.
# Files 01-08 run in order in one namespace, like notebook cells (see run.py).

# Voice spoofing detector: frozen XLS-R 300M + LoRA, weighted hidden states,
# master-token attention pooling, OC-Softmax with SupCon and evidential auxiliary losses.
# Trained on ASVspoof 2019 LA (train), evaluated without fine-tuning on
# ASVspoof 2019 LA (eval), ASVspoof 2021 LA (eval), In-the-Wild and ASVspoof 5 (eval).



# %% ======================== Cell 1. Configuration ========================
import os
import random
import glob
import subprocess
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count()

# Data paths
ROOT = "/kaggle/input/datasets/illrazu/la-asvspoof/LA/LA"
TRAIN_FLAC_DIR = f"{ROOT}/ASVspoof2019_LA_train/flac"
DEV_FLAC_DIR = f"{ROOT}/ASVspoof2019_LA_dev/flac"
EVAL_FLAC_DIR = f"{ROOT}/ASVspoof2019_LA_eval/flac"
PROTOCOL_DIR = f"{ROOT}/ASVspoof2019_LA_cm_protocols"
TRAIN_PROTOCOL = f"{PROTOCOL_DIR}/ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = f"{PROTOCOL_DIR}/ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTOCOL = f"{PROTOCOL_DIR}/ASVspoof2019.LA.cm.eval.trl.txt"

DATASET_2021 = "/kaggle/input/datasets/illrazu/la-asvspoof-2021/2021"
FLAC_2021 = os.path.join(DATASET_2021, "flac")
KEY_FILE_2021 = os.path.join(DATASET_2021, "LA-keys-full/keys/LA/CM/trial_metadata.txt")

ITW_ROOT = "/kaggle/input/datasets/illrazu/in-the-wild/release_in_the_wild"
ITW_META = os.path.join(ITW_ROOT, "meta.csv")

ASV5_PROTO_DIR = "/kaggle/input/datasets/trungkinenguyn/asvspoof5-protocols"
ASV5_AUDIO_DIR = "/kaggle/input/datasets/trungkinenguyn/asvspoof5"

WAV2VEC_MODEL_PATH = "/kaggle/input/datasets/illrazu/code-for-kaggle-arc/wav2vec2-xls-r-300m"
OUT_DIR = "/kaggle/working"

# Input
SAMPLE_RATE = 16000
MAX_LEN_SEC = 4.0
MAX_LEN = int(SAMPLE_RATE * MAX_LEN_SEC)

# Training (effective batch: 48 on one GPU, 64 on two GPUs)
BATCH_SIZE = 16 * max(1, N_GPUS)
GRAD_ACCUM = 3 if N_GPUS <= 1 else 2
NUM_WORKERS_TRAIN = 4
EPOCHS = 10
LR_HEAD = 1e-4
LR_LORA = 5e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

# Model
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
PROJ_DIM = 256
LAYER_SUBSET = list(range(5, 9)) + list(range(16, 21))   # prior emphasis on layers 5-8 and 16-20
OC_ALPHA = 20.0
OC_M_REAL = 0.7
OC_M_FAKE = 0.2

# Loss weights
CONTRASTIVE_LOSS_WEIGHT = 0.3
EVIDENTIAL_LOSS_WEIGHT = 0.05

# Augmentation probabilities
RAWBOOST_PROB = 0.7
TELEPHONE_PROB = 0.5
WILD_PROB = 0.5

RUN_NAME = "G_rawboost_fix"
BEST_CKPT_PATH = f"{OUT_DIR}/best_model_{RUN_NAME}.pt"     # lowest dev EER
FINAL_CKPT_PATH = f"{OUT_DIR}/final_model_{RUN_NAME}.pt"   # after the last epoch
RESUME_PATH = f"{OUT_DIR}/last_{RUN_NAME}.pt"

assert subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0, "ffmpeg is required"
