import random
from pathlib import Path

import torch

AUDIO_DIR = Path("app/audio/1350")
SECONDARY_AUDIO_DIRS = [Path("app/audio/strong"), Path("app/audio/night")]
CHANNELS = 1
SAMPLE_RATE = 22050
RECORD_SAMPLE_RATE = 44100
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WINDOW_TIME = 0.2  # In seconds

DTYPE_RECORD = "int16"
DTYPE_PROCESS = "float32"

OUTPUT_FILENAME = "background"  # Must not include the extension
TIME_SKIP = 0.33  # Time to remove from start and end of recording.

NOISE_FILENAME = "background.wav"
BLACKLIST = ["background"]

MODEL_OUTPUT_DIR = AUDIO_DIR / "models"
MODEL_PATH = MODEL_OUTPUT_DIR / "keystroke_classifier.pth"
METADATA_PATH = MODEL_OUTPUT_DIR / "model_metadata.json"
DATASET_PATH = AUDIO_DIR / "dataset"
RIR_PATH = Path("app/audio/rir/MIT")

TOGGLE_KFOLD = False
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 5
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = random.randint(0, 10_000)  # noqa: S311
