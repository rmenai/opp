"""
This script records audio upon a key press, processes the audio,
and uses a pre-trained model to predict the key.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import onnxruntime
import torch

CHANNELS = 1
DTYPE_RECORD = "int16"
DTYPE_PROCESS = "float32"

SAMPLE_RATE = 44100
N_MELS = 64  # This will be overridden by metadata if available
N_FFT = 1024
HOP_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = Path("audio/1350/models")

MODEL_PATH = MODEL_DIR / "keystroke_classifier.onnx"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def compute_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
):
    """Computes a log-Mel spectrogram from an audio signal."""
    s = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_s = librosa.power_to_db(s, ref=np.max)
    return log_s.astype(np.float32)


# KeystrokeCNN class is no longer needed as we are using ONNX


def load_model_and_metadata():
    """Load the ONNX model and its metadata."""
    # Load metadata
    with Path.open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Load ONNX model
    ort_session = onnxruntime.InferenceSession(str(MODEL_PATH))

    # Ensure metadata n_mels is used if available, otherwise keep global
    global N_MELS
    if "n_mels" in metadata:
        N_MELS = metadata["n_mels"]
    else:
        # Fallback or error if n_mels not in metadata and critical
        print(f"Warning: 'n_mels' not found in metadata, using global N_MELS={N_MELS}")

    return ort_session, metadata


# def predict_keystroke(signal: np.ndarray, ort_session: onnxruntime.InferenceSession, metadata: dict):
#     """
#     Predict the keystroke from an audio signal using ONNX model.
#
#     Args:
#         signal: Raw audio signal as a numpy array.
#         ort_session: Loaded ONNX runtime inference session.
#         metadata: Dictionary containing model metadata (n_mels, time_bins, key_map).
#
#     Returns:
#         tuple: (predicted_key, confidence_score, all_probabilities)
#     """
#     key_map = metadata["key_map"]
#     target_n_mels = metadata["n_mels"]
#     target_time_bins = metadata["time_bins"]
#
#     # Convert signal to mel spectrogram
#     # Use n_mels from metadata for consistency
#     mel_spec = compute_log_mel_spectrogram(
#         signal,
#         sr=SAMPLE_RATE,
#         n_mels=target_n_mels,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#     )
#
#     # Pad or truncate the spectrogram to match the expected time_bins
#     current_n_mels, current_time_bins = mel_spec.shape


def predict_keystroke(signal: np.ndarray, ort_session: onnxruntime.InferenceSession, metadata: dict):
    """
    Predict the keystroke from an audio signal using ONNX model.

    Args:
        signal: Raw audio signal as a numpy array.
        ort_session: Loaded ONNX runtime inference session.
        metadata: Dictionary containing model metadata (n_mels, time_bins, key_map).

    Returns:
        tuple: (predicted_key, confidence_score, all_probabilities)
    """
    key_map = metadata["key_map"]
    target_n_mels = metadata["n_mels"]
    target_time_bins = metadata["time_bins"]

    # Convert signal to mel spectrogram
    # Use n_mels from metadata for consistency
    mel_spec = compute_log_mel_spectrogram(
        signal,
        sr=SAMPLE_RATE,
        n_mels=target_n_mels,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # Pad or truncate the spectrogram to match the expected time_bins
    current_n_mels, current_time_bins = mel_spec.shape

    if current_n_mels != target_n_mels:
        # This should ideally not happen if compute_log_mel_spectrogram uses target_n_mels
        raise ValueError(
            f"Mel spectrogram n_mels ({current_n_mels}) does not match model's expected n_mels ({target_n_mels})",
        )

    if current_time_bins < target_time_bins:
        padding_width = target_time_bins - current_time_bins
        # Pad on the right side (axis=1) with a value representing silence
        mel_spec = np.pad(
            mel_spec,
            ((0, 0), (0, padding_width)),
            mode="constant",
            constant_values=np.log(1e-10),
        )  # Approx -80dB for silence
    elif current_time_bins > target_time_bins:
        # Truncate from the right side (axis=1)
        mel_spec = mel_spec[:, :target_time_bins]

    # Prepare input for ONNX model
    # Expected shape: (batch_size, channels, n_mels, time_bins)
    input_tensor = mel_spec.reshape(1, 1, target_n_mels, target_time_bins).astype(np.float32)

    # Make prediction
    ort_inputs = {"input": input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)  # Using None for output names to get all outputs
    logits = ort_outs[0]  # Assuming the first output is the logits

    # Process logits
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1)
    predicted_class_idx = torch.argmax(torch.from_numpy(logits), dim=1).item()
    confidence = probabilities[0, predicted_class_idx].item()

    # Convert class index to key name
    inv_key_map = {v: k for k, v in key_map.items()}
    predicted_key = inv_key_map[predicted_class_idx]

    return predicted_key, confidence, probabilities[0].cpu().numpy()


# def load_model_and_metadata():
#     """Load the trained model and its metadata."""
#     # Load metadata
#     with Path.open(METADATA_PATH) as f:
#         metadata = json.load(f)
#
#     # Load model
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
#     model = KeystrokeCNN(
#         n_mels=metadata["n_mels"],
#         time_bins=metadata["time_bins"],
#         num_classes=metadata["num_classes"],
#     )
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(DEVICE)
#     model.eval()
#
#     return model, metadata


# def predict_keystroke(signal: np.ndarray, model, key_map: dict):
#     """
#     Predict the keystroke from a preprocessed audio signal.
#
#     Args:
#         signal: Preprocessed audio signal (normalized, noise reduced, correct size)
#         model: Loaded PyTorch model
#         key_map: Dictionary mapping class indices to key names
#
#     Returns:
#         tuple: (predicted_key, confidence_score, all_probabilities)
#     """
#     # Convert signal to mel spectrogram
#     mel_spec = compute_log_mel_spectrogram(signal)
#
#     # Convert to tensor and add batch and channel dimensions
#     input_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).to(DEVICE)
#
#     # Make prediction
#     with torch.no_grad():
#         logits = model(input_tensor)
#         probabilities = torch.nn.functional.softmax(logits, dim=1)
#         predicted_class = torch.argmax(logits, dim=1).item()
#         confidence = probabilities[0, predicted_class].item()
#
#     # Convert class index to key name
#     predicted_key = key_map[str(predicted_class)]
#
#     return predicted_key, confidence, probabilities[0].cpu().numpy()


def main():
    """Example usage of the keystroke prediction."""
    # Load model and metadata
    print("Loading model...")
    model, metadata = load_model_and_metadata()
    key_map = metadata["key_map"]
    inv_key_map = {v: k for k, v in key_map.items()}

    print("Model loaded successfully!")
    print(f"Available keys: {list(key_map.values())}")

    # Example: Load your preprocessed signal
    # Replace this with your actual preprocessed signal
    # signal = your_preprocessed_signal  # This should be a numpy array

    # For demonstration, create a dummy signal
    # Remove this and use your actual signal
    signal = np.random.randn(22050).astype(np.float32)  # 0.5 seconds at 44.1kHz

    # Make prediction
    predicted_key, confidence, all_probs = predict_keystroke(signal, model, metadata)

    print(f"\nPredicted key: {predicted_key}")
    print(f"Confidence: {confidence:.4f}")

    # Show top 3 predictions
    sorted_indices = np.argsort(all_probs)[::-1]
    print("\nTop 3 predictions:")
    for i in range(min(3, len(sorted_indices))):
        idx = sorted_indices[i]
        key = inv_key_map[idx]
        prob = all_probs[idx]
        print(f"  {i + 1}. {key}: {prob:.4f}")


if __name__ == "__main__":
    main()
