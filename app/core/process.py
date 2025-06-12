"""
This script processes raw audio files for keystroke detection. It performs noise reduction,
normalization, event-based audio segmentation, and computes log-mel spectrograms for each
keystroke event. The processed data is saved as isolated audio snippets and a compressed
dataset suitable for machine learning tasks.
"""

import json
import time
from itertools import pairwise

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

from app.core.config import (
    AUDIO_DIR,
    BLACKLIST,
    DTYPE_PROCESS,
    DTYPE_RECORD,
    NOISE_FILENAME,
    SAMPLE_RATE,
    SECONDARY_AUDIO_DIRS,
    WINDOW_TIME,
)


def main():
    AUDIO_DIRS = [AUDIO_DIR, *SECONDARY_AUDIO_DIRS]

    samples = []
    labels = []

    dataset_dir = AUDIO_DIR / "dataset"

    for audio_dir in AUDIO_DIRS:
        raw_dir = audio_dir / "raw"
        clean_dir = audio_dir / "clean"
        iso_dir = audio_dir / "isolated"
        aug_dir = audio_dir / "augmented"

        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        iso_dir.mkdir(parents=True, exist_ok=True)
        aug_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        noise_data, sr = sf.read(raw_dir / NOISE_FILENAME, dtype=DTYPE_RECORD)
        noise_data = librosa.resample(y=noise_data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)

        # Load each audio file.
        for wav_path in sorted(raw_dir.glob("*.wav")):
            stem = wav_path.stem
            if stem in BLACKLIST:
                continue

            data = json.load((raw_dir / f"{stem}.json").open())
            y, sr = sf.read(wav_path, dtype=DTYPE_RECORD)
            y = librosa.resample(y=y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)

            print("Performing noise reduction...")
            y_clean = nr.reduce_noise(y=y, sr=SAMPLE_RATE, y_noise=noise_data)
            print("Noise reduction complete.")

            print("Performing normalization...")
            y_clean /= np.max(np.abs(y_clean))
            print("Normalization complete.")

            out_path = clean_dir / f"{stem}.wav"
            sf.write(out_path, y_clean, SAMPLE_RATE, subtype="FLOAT")
            print("Saved clean file.")

            events = data["keystrokes"]
            start_time = data.get("start_time", 0)
            window = int(WINDOW_TIME * SAMPLE_RATE)
            idx = 0
            for pressed, released in pairwise(events):
                if pressed["value"] == 1 and released["value"] == 0 and pressed["code"] == released["code"]:
                    t_center = (pressed["timestamp"] + released["timestamp"]) / 2 - start_time
                    center_sample = int(t_center * SAMPLE_RATE)
                    start = max(0, center_sample - window // 2)
                    end = min(len(y_clean), center_sample + window // 2)
                    snippet = y_clean[start:end]

                    # Save isolated wav
                    iso_path = iso_dir / f"{stem}.{idx:03d}.wav"
                    sf.write(iso_path, snippet, SAMPLE_RATE, subtype="FLOAT")
                    samples.append(snippet)
                    labels.append(stem)
                    idx += 1

    # Encode labels
    unique_keys = sorted(set(labels))
    key_map = {k: i for i, k in enumerate(unique_keys)}
    y_labels = np.array([key_map[k] for k in labels], dtype=np.int16)

    # Stack signals (pad/truncate to same length)
    max_len = max(sig.shape[-1] for sig in samples)
    x = np.zeros((len(samples), max_len), dtype=DTYPE_PROCESS)
    for i, sig in enumerate(samples):
        sig_len = sig.shape[-1]
        if max_len > sig_len:
            x[i, :sig_len] = sig
        else:
            x[i, :] = sig[:max_len]

    timestamp = int(time.time())
    dataset_filename = f"keys_{timestamp}_{len(labels)}.npz"

    # Save dataset
    np.savez_compressed(
        dataset_dir / dataset_filename,
        signals=x,
        labels=y_labels,
        label_map=json.dumps(key_map),
    )

    print(f"Saved dataset with {len(samples)} samples to '{dataset_filename}'")
