import sys
import threading
import time

import librosa
import noisereduce
import numpy as np
import sounddevice as sd
import tensorflow as tf
from pynput import keyboard

# --- Constants derived from record.py, process.py, train.py ---
SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_FILENAME = "audio_command_model.keras"  # Ensure this model is accessible

# Processing constants from process.py
MAX_AUDIO_LEN_SECONDS = 1.0  # Max duration of audio to process
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
TRIM_TOP_DB = 20
NOISE_REDUCTION_PROP_DECREASE = 1.0

# Calculated max length for MFCC features
# This should match the training configuration
MAX_LEN_SAMPLES = int(MAX_AUDIO_LEN_SECONDS * SAMPLE_RATE)
MAX_LEN_MFCC = int(np.ceil(MAX_LEN_SAMPLES / HOP_LENGTH))  # Should be 32 for 1s audio @ 16kHz, hop 512

# Labels from train.py (ensure this order matches model output)
LABELS = [
    "background",
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]

# --- Global variables for recording and keypress ---
recorded_frames = []
stop_recording_flag = threading.Event()
user_wants_to_quit = False
keypress_detected_char = None


# --- Audio Processing Functions (adapted from process.py) ---
def trim_audio_data(audio_data, sr, top_db=TRIM_TOP_DB):
    """Trims silence from the beginning and end of audio data."""
    if len(audio_data) == 0:
        return audio_data
    trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=top_db)
    return trimmed_audio


def reduce_noise_data(audio_data, sr, prop_decrease=NOISE_REDUCTION_PROP_DECREASE):
    """Reduces noise in audio data."""
    if len(audio_data) == 0:
        return audio_data
    # Ensure audio_data is float for noisereduce
    audio_data_float = audio_data.astype(np.float32)
    reduced_noise_audio = noisereduce.reduce_noise(y=audio_data_float, sr=sr, prop_decrease=prop_decrease)
    return reduced_noise_audio


def extract_features_from_audio(audio_data, sr):
    """Extracts, pads/truncates, and normalizes MFCC features from audio data."""
    if len(audio_data) == 0:
        return np.zeros((N_MFCC, MAX_LEN_MFCC))  # Return empty/zero features

    # Ensure audio is long enough for at least one frame, or handle gracefully
    if len(audio_data) < N_FFT:
        # Pad with zeros if too short for one FFT window
        audio_data = np.pad(audio_data, (0, N_FFT - len(audio_data)), "constant")

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Pad or truncate MFCCs to MAX_LEN_MFCC
    if mfccs.shape[1] < MAX_LEN_MFCC:
        pad_width = MAX_LEN_MFCC - mfccs.shape[1]
        mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs_padded = mfccs[:, :MAX_LEN_MFCC]

    # Normalize MFCCs (feature-wise)
    if np.std(mfccs_padded) != 0:
        mfccs_normalized = (mfccs_padded - np.mean(mfccs_padded)) / np.std(mfccs_padded)
    else:  # Avoid division by zero if std is zero (e.g. silent input)
        mfccs_normalized = mfccs_padded - np.mean(mfccs_padded)

    return mfccs_normalized


# --- Model Loading ---
def load_trained_model(model_path):
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


# --- Keyboard Listener Callback ---
def on_press(key):
    global stop_recording_flag, user_wants_to_quit, keypress_detected_char
    try:
        keypress_detected_char = key.char
        if key.char == "q":
            user_wants_to_quit = True
    except AttributeError:  # Special keys (like shift, ctrl, etc.)
        keypress_detected_char = None  # Or handle special keys if needed
        # Or treat any keypress (non-'q') as signal to process

    stop_recording_flag.set()  # Signal to stop recording
    return False  # Stop the listener


# --- Main Prediction Loop ---
def predict_live():
    global recorded_frames, stop_recording_flag, user_wants_to_quit, keypress_detected_char

    model = load_trained_model(MODEL_FILENAME)

    while not user_wants_to_quit:
        recorded_frames = []
        stop_recording_flag.clear()
        keypress_detected_char = None

        print("\nPress any key to stop recording and predict. Press 'q' to quit.")

        # Define callback for sounddevice InputStream
        def audio_callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            recorded_frames.append(indata.copy())

        # Start listener in a separate thread
        listener_thread = keyboard.Listener(on_press=on_press)
        listener_thread.start()

        # Start recording
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
        with stream:
            print("🔴 Recording...")
            stop_recording_flag.wait()  # Wait until a key is pressed

        listener_thread.join()  # Ensure listener thread has finished

        if user_wants_to_quit and keypress_detected_char == "q":
            print("Exiting...")
            break

        print("Recording stopped.")

        if not recorded_frames:
            print("No audio recorded.")
            continue

        # Concatenate recorded frames
        audio_data = np.concatenate(recorded_frames, axis=0)
        if audio_data.ndim > 1 and audio_data.shape[1] == CHANNELS:
            audio_data = audio_data.flatten()  # Ensure mono

        # 1. Trim silence
        print("Trimming audio...")
        audio_trimmed = trim_audio_data(audio_data, SAMPLE_RATE)
        if len(audio_trimmed) == 0:
            print("Audio is all silence after trimming.")
            continue

        # 2. Reduce noise
        print("Reducing noise...")
        audio_denoised = reduce_noise_data(audio_trimmed, SAMPLE_RATE)

        # 3. Extract features (MFCC, padding, normalization)
        print("Extracting features...")
        features = extract_features_from_audio(audio_denoised, SAMPLE_RATE)

        # Reshape features for the model: (1, num_features_time_steps, num_mfcc_coeffs)
        # The features are (N_MFCC, MAX_LEN_MFCC), need (MAX_LEN_MFCC, N_MFCC) for model if channels_last
        # Or (N_MFCC, MAX_LEN_MFCC) if model expects (batch, features, timesteps)
        # Keras Conv1D/LSTM usually expect (batch, timesteps, features)
        # Our MFCCs are (n_mfcc, time_frames), so transpose.
        features_for_model = features.T
        features_for_model = np.expand_dims(features_for_model, axis=0)  # Add batch dimension

        # 4. Predict
        print("Predicting...")
        prediction = model.predict(features_for_model)
        predicted_index = np.argmax(prediction[0])
        predicted_label = LABELS[predicted_index]
        confidence = prediction[0][predicted_index]

        print(f"✅ Predicted command: '{predicted_label}' (Confidence: {confidence:.2f})")

        # Brief pause before next cycle
        time.sleep(1)


if __name__ == "__main__":
    print("Starting live prediction script...")
    print(f"Audio settings: SR={SAMPLE_RATE}Hz, Channels={CHANNELS}")
    print(f"Feature settings: MFCCs={N_MFCC}, MaxLenMFCC={MAX_LEN_MFCC}")
    print(f"Using model: {MODEL_FILENAME}")
    print(f"Available commands: {len(LABELS)}")
    try:
        predict_live()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Script finished.")
