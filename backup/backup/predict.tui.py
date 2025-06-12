"""
This script records audio upon a key press, processes the audio,
and uses a pre-trained model to predict the key.
"""

import asyncio
import json
import time
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from config import (
    AUDIO_DIR,
    DEVICE,
    DTYPE_RECORD,
    METADATA_PATH,
    MODEL_PATH,
    NOISE_FILENAME,
    SAMPLE_RATE,
    WINDOW_TIME,
)
from evdev import InputDevice, ecodes
from record import AudioRecorder
from train import KeystrokeCNN
from utils import compute_log_mel_spectrogram

mpl.use("TkAgg")
dev = InputDevice("/dev/input/event0")


def plot_time(signal: np.ndarray, name: str):
    """Plot the signal in the time domain."""
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(f"Time Domain Signal: {name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def load_model_and_metadata():
    """Load the trained model and its metadata."""
    with Path.open(METADATA_PATH) as f:
        metadata = json.load(f)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = KeystrokeCNN(
        n_mels=metadata["n_mels"],
        time_bins=metadata["time_bins"],
        num_classes=metadata["num_classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, metadata


def predict_keystroke(signal: np.ndarray, model, key_map: dict) -> tuple:  # noqa: ANN001
    """
    Predict the keystroke from a preprocessed audio signal.

    Args:
        signal: Preprocessed audio signal (normalized, noise reduced, correct size)
        model: Loaded PyTorch model
        key_map: Dictionary mapping class indices to key names

    Returns:
        tuple: (predicted_key, confidence_score, all_probabilities)
    """
    mel_spec = compute_log_mel_spectrogram(signal)
    print(mel_spec.shape)
    input_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    predicted_key = key_map[predicted_class]
    return predicted_key, confidence, probabilities[0].cpu().numpy()


async def main():
    """Example usage of the keystroke prediction."""
    print("Loading model...")
    model, metadata = load_model_and_metadata()
    key_map = {v: k for k, v in metadata["key_map"].items()}

    print("Model loaded successfully!")
    print(f"Available keys: {list(key_map.values())}")

    raw_dir = AUDIO_DIR / "raw"
    window = int(WINDOW_TIME * SAMPLE_RATE)

    noise_data, sr = sf.read(raw_dir / NOISE_FILENAME, dtype=DTYPE_RECORD)
    noise_data = librosa.resample(y=noise_data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)

    while True:
        recorder = AudioRecorder("", 0)
        is_recording = True
        recorder.start()
        time.sleep(1)

        # noise_data = recorder.save()[window:]

        print("Press 'ESC' to stop.")
        event_time = time.time()
        toggle = False

        async for ev in dev.async_read_loop():
            if not is_recording:
                is_recording = True
                recorder.start()

            if ev.type == ecodes.EV_KEY:  # Is not <ESC>.
                if ev.code == ecodes.KEY_ESC and ev.value == 1:
                    break

                if ev.value == 1:
                    event_time = time.time()
                    toggle = True
                    print(f"Key pressed at {event_time}")
                elif toggle:
                    event_time = (event_time + time.time()) / 2
                    toggle = False
                    # break

        time.sleep(0.5)

        y, sr = recorder.stop()
        y = librosa.resample(y=y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)

        y_clean = nr.reduce_noise(y=y, sr=SAMPLE_RATE, y_noise=noise_data)
        y_clean /= np.max(np.abs(y_clean))

        t = event_time - recorder.start_time
        center_sample = int(t * SAMPLE_RATE)
        start = max(0, center_sample - window // 2)
        end = min(len(y_clean), center_sample + window // 2)
        snippet = y_clean[start:end]

        print(snippet.shape)
        predicted_key, confidence, all_probs = predict_keystroke(snippet, model, key_map)

        print(f"\nPredicted key: {predicted_key}")
        print(f"Confidence: {confidence:.4f}")

        sorted_indices = np.argsort(all_probs)[::-1]
        print("\nTop 3 predictions:")
        for i in range(min(3, len(sorted_indices))):
            idx = sorted_indices[i]
            key = key_map[idx]
            prob = all_probs[idx]
            print(f"  {i + 1}. {key}: {prob:.4f}")

        plot_time(snippet, f"{predicted_key}, {confidence:.4f}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting program.")
