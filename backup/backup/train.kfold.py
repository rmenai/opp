"""
This script trains and evaluates a Convolutional Neural Network (CNN)
for classifying keystroke audio signals using k-fold cross-validation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    ApplyImpulseResponse,
    Compose,
    Gain,
    PitchShift,
    Shift,
    TimeMask,
    TimeStretch,
)
from config import (
    AUDIO_DIR,
    BATCH_SIZE,
    DATASET_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_OUTPUT_DIR,
    N_SPLITS,
    PATIENCE,
    RIR_PATH,
    SAMPLE_RATE,
    SEED,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import compute_log_mel_spectrogram, set_seed


class KeystrokeDataset(Dataset):
    """Custom PyTorch Dataset for keystroke audio."""

    def __init__(self, features: list, labels: list, augmentations=None) -> None:  # noqa: ANN001
        self.features = features
        self.labels = torch.from_numpy(np.array(labels)).long()
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> (list, list):
        x = self.features[idx]
        y = self.labels[idx]

        if self.augmentations:
            x = self.augmentations(samples=x, sample_rate=SAMPLE_RATE)

        x = compute_log_mel_spectrogram(x)
        x = torch.from_numpy(x).unsqueeze(0)  # Add channel dimension

        return x, y


class KeystrokeCNN(nn.Module):
    """A simple CNN for classifying keystroke spectrograms."""

    def __init__(self, n_mels: int, time_bins: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_mels, time_bins)
            feat_dim = self.features(dummy_input).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
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


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
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


def main():
    """Main function to run the training and evaluation pipeline."""
    set_seed(SEED)
    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        recent_path = sorted(
            DATASET_PATH.glob("keys_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[0]
    except IndexError:
        print(f"Error: No dataset files found in {DATASET_PATH}")
        return

    print(f"Loading data from most recent dataset: {recent_path.name}")
    npz = np.load(recent_path, allow_pickle=True)

    X, y = npz["signals"], npz["labels"]
    key_map = json.loads(npz["label_map"].item())
    num_classes = len(key_map)

    raw_dir = AUDIO_DIR / "raw"

    signal_augmentations = Compose(
        [
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.4),
            TimeMask(min_band_part=0.0, max_band_part=0.05, p=0.2),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
            Shift(min_shift=-0.3, max_shift=0.4, p=0.3),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.4),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            AddBackgroundNoise(raw_dir / "background.wav", p=0.1),
            ApplyImpulseResponse(ir_path=RIR_PATH, p=0.3),
        ],
    )

    # --- K-Fold Cross-Validation Setup ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== Fold {fold + 1}/{N_SPLITS} =====\n")

        # --- Create Datasets and Dataloaders for the current fold ---
        train_dataset = KeystrokeDataset(X[train_idx], y[train_idx], augmentations=signal_augmentations)
        val_dataset = KeystrokeDataset(X[val_idx], y[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # --- Model, Loss, Optimizer, and Scheduler ---
        dummy_spec, _ = train_dataset[0]
        _, n_mels, time_bins = dummy_spec.shape
        model = KeystrokeCNN(n_mels, time_bins, num_classes).to(DEVICE)
        print(f"Model initialized on {DEVICE}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
            patience=3,
        )

        # --- Training Loop with Early Stopping ---
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_path = MODEL_OUTPUT_DIR / f"cnn_best_fold{fold + 1}_{SEED}.pth"

        for epoch in range(1, EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Validation loss improved. Saving model to {best_model_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

        # --- Evaluate the best model for the fold on the validation set ---
        print(f"\nLoading best model for fold {fold + 1} for final evaluation...")
        model.load_state_dict(torch.load(best_model_path))
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        fold_results.append({"val_loss": val_loss, "val_acc": val_acc})

        print(f"\n--- Fold {fold + 1} Results ---")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

    # --- Aggregate and Display Final Results ---
    avg_val_loss = np.mean([res["val_loss"] for res in fold_results])
    avg_val_acc = np.mean([res["val_acc"] for res in fold_results])
    std_val_acc = np.std([res["val_acc"] for res in fold_results])

    print("\n\n--- K-Fold Cross-Validation Summary ---")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f} (+/- {std_val_acc:.4f})")

    # Optionally, you can now train a final model on the full dataset and save it
    # This part is similar to your original script's final steps
    print("\nTraining final model on the entire dataset...")
    full_dataset = KeystrokeDataset(X, y, augmentations=signal_augmentations)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Re-initialize the model
    final_model = KeystrokeCNN(n_mels, time_bins, num_classes).to(DEVICE)
    final_optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    final_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Final Training Epoch {epoch}/{EPOCHS} ---")
        train_one_epoch(final_model, full_loader, final_criterion, final_optimizer, DEVICE)

    print("Final model training complete.")

    final_model_export_path = MODEL_OUTPUT_DIR / "keystroke_classifier_final.pth"
    onnx_export_path = MODEL_OUTPUT_DIR / "keystroke_classifier.onnx"
    metadata_path = MODEL_OUTPUT_DIR / "model_metadata.json"

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "n_mels": n_mels,
            "time_bins": time_bins,
            "num_classes": num_classes,
            "key_map": key_map,
        },
        final_model_export_path,
    )
    print(f"Final model exported for inference to: {final_model_export_path}")

    dummy_input = torch.randn(1, 1, n_mels, time_bins, device=next(final_model.parameters()).device)
    torch.onnx.export(
        final_model,
        dummy_input,
        onnx_export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported in ONNX format to: {onnx_export_path}")

    with Path.open(metadata_path, "w") as f:
        json.dump(
            {
                "n_mels": n_mels,
                "time_bins": time_bins,
                "num_classes": num_classes,
                "key_map": key_map,
            },
            f,
            indent=4,
        )
    print(f"Model metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
