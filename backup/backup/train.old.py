"""
This script trains a convolutional neural network (CNN) to classify keystroke audio signals.
It loads preprocessed datasets, applies data augmentation, splits data into train/val/test sets,
defines a CNN model, and runs training with early stopping and evaluation.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from audiomentations import (
    AddGaussianNoise,
    ApplyImpulseResponse,
    Compose,
    Gain,
    PitchShift,
    Shift,
    TimeMask,
    TimeStretch,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

AUDIO_DIR = Path("audio/1350")
WINDOW_TIME = 0.33  # In seconds
CHANNELS = 1

SAMPLE_RATE = 44100
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


def compute_log_mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_s = librosa.power_to_db(s, ref=np.max)
    return log_s.astype(np.float32)


def set_seed(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


# --- 1) Data Augmentation & Custom Dataset ---
class KeystrokeDataset(Dataset):
    """Custom PyTorch Dataset for keystroke spectrograms."""

    def __init__(self, features, labels, augmentations=None):
        self.features = features
        self.labels = torch.from_numpy(labels).long()
        self.augmentations = augmentations

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.augmentations:
            x = self.augmentations(samples=x, sample_rate=SAMPLE_RATE)

        x = compute_log_mel_spectrogram(x, SAMPLE_RATE)
        x = torch.from_numpy(x).unsqueeze(0)

        return x, y


# --- 2) Model Definition ---
class KeystrokeCNN(nn.Module):
    """A simple CNN for classifying keystroke spectrograms."""

    def __init__(self, n_mels, time_bins, num_classes):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> (16, n_mels, time_bins)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, n_mels/2, time_bins/2)
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (32, n_mels/2, time_bins/2)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, n_mels/4, time_bins/4)
            nn.BatchNorm2d(32),
        )

        # Dynamically calculate the flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_mels, time_bins)
            feat_dim = self.features(dummy_input).view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --- 3) Training and Evaluation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(dataloader, desc="Training"):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc="Evaluating"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(yb.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    return avg_loss, accuracy, np.concatenate(all_labels), np.concatenate(all_preds)


# --- 4) Main Execution Block ---
def main(args):
    """Main function to run the training and evaluation pipeline."""
    set_seed(args.seed)

    # --- Data Loading and Splitting ---
    recent_path = sorted(
        [f for f in os.listdir(str(AUDIO_DIR / "dataset")) if f.endswith(".npz") and f.startswith("keys_")],
        reverse=True,
    )[0]

    logging.info(f"Loading data from most recent dataset, {recent_path}")
    npz = np.load(AUDIO_DIR / "dataset" / recent_path, allow_pickle=True)

    X, y = npz["specs"], npz["labels"]
    key_map = json.loads(npz["label_map"].item())
    num_classes = len(key_map)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=args.seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        random_state=args.seed,
        stratify=y_tmp,
    )
    logging.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    signal_augmentations = Compose(
        [
            # Time & Pitch
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.4),
            TimeMask(min_band_part=0.0, max_band_part=0.05, fade_duration=0.05, p=0.2),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
            Shift(min_shift=-0.4, max_shift=0.4, rollover=False, p=0.3),
            Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.4),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
            ApplyImpulseResponse(
                ir_path=Path("audio/rir/MIT"),
                p=0.3,
            ),
        ],
    )

    # --- Augmentations and DataLoaders ---
    train_dataset = KeystrokeDataset(
        X_train,
        y_train,
        augmentations=signal_augmentations,
    )
    val_dataset = KeystrokeDataset(X_val, y_val)  # No augmentation for validation/test
    test_dataset = KeystrokeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # --- Model, Loss, Optimizer, and Scheduler ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dummy_spec, _ = train_dataset[0]
    _, n_mels, time_bins = dummy_spec.shape
    model = KeystrokeCNN(n_mels, time_bins, num_classes).to(device)
    logging.info(f"Model initialized on {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=3)

    # --- Training Loop with Early Stopping ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, f"cnn_best_{args.seed}.pth")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)  # LR scheduler step

        logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Validation loss improved. Saving model to {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= args.patience:
            logging.info("Early stopping triggered.")
            break

    # --- Final Test Evaluation ---
    logging.info("Loading best model for final test evaluation.")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, device)

    logging.info("\nFinal Test Results:")
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:\n", cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Keystroke Classification CNN")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=AUDIO_DIR / "models",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=random.randint(0, 10000), help="Random seed for reproducibility")

    args = parser.parse_args()
    main(args)
