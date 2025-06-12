import sys
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter

mpl.use("TkAgg")

AUDIO_DIR = Path("audio")
DEFAULT_SR = 44100
FRAME_SIZE = 2048
HOP_LENGTH = 512

KEYSTROKE_LENGTH = 14400
TRIGGER_LEVEL = 5.0
RELEASE_LEVEL = 1.5
COOLDOWN_MS = 100


if not AUDIO_DIR.exists():
    print(f"Audio directory not found at: {AUDIO_DIR.absolute()}")
    print("Please create it and place your .wav files inside.")
    sys.exit()


def highpass_filter(y: np.ndarray, sr: int, cutoff: int = 500) -> np.ndarray:
    """Applies a high-pass filter to remove low-frequency noise/hum."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(5, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, y)


def isolate_keystrokes_rolling_window(
    y: np.ndarray,
    sr: int,
    keystroke_len: int = KEYSTROKE_LENGTH,
    trigger_level: float = TRIGGER_LEVEL,
    release_level: float = RELEASE_LEVEL,
    cooldown_ms: int = COOLDOWN_MS,
) -> list[np.ndarray]:
    """
    Isolates keystrokes using a stable, rolling-window energy-based method.

    Args:
        y: The input audio time series.
        sr: The sampling rate of y.
        keystroke_len: The fixed number of samples for each extracted keystroke.
        trigger_level: How many times louder than the average noise a sound must be to start an event.
        release_level: How many times louder than the average noise a sound must be to be considered "over".
        cooldown_ms: How many milliseconds to wait after an event ends before looking for a new one.
    """
    # 1. Pre-process the audio to remove noise
    y_filtered = highpass_filter(y, sr)

    # 2. Calculate rolling RMS energy
    hop_length = HOP_LENGTH
    rms = librosa.feature.rms(y=y_filtered, frame_length=FRAME_SIZE, hop_length=hop_length)[0]

    # 3. Define thresholds based on the audio's energy profile
    noise_floor_rms = np.median(rms)  # Use median for a robust estimate of "silence"
    on_threshold = noise_floor_rms * trigger_level
    off_threshold = noise_floor_rms * release_level

    # 4. Implement the two-threshold state machine with a cooldown
    onsets = []
    state = "IDLE"  # Can be "IDLE" or "TRIGGERED"
    cooldown_frames = librosa.time_to_frames(cooldown_ms / 1000, sr=sr, hop_length=hop_length)
    cooldown_counter = 0

    for i in range(len(rms)):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue

        if state == "IDLE" and rms[i] > on_threshold:
            # Event starts!
            start_sample = librosa.frames_to_samples(i, hop_length=hop_length)
            onsets.append(start_sample)
            state = "TRIGGERED"

        elif state == "TRIGGERED" and rms[i] < off_threshold:
            # Event ends, start the cooldown
            state = "IDLE"
            cooldown_counter = cooldown_frames

    # 5. Extract fixed-length chunks from the original audio
    extracted_keys = []
    for start_sample in onsets:
        end_sample = start_sample + keystroke_len
        if end_sample < len(y):
            keystroke = y[start_sample:end_sample]
            extracted_keys.append(keystroke)

    return extracted_keys


def play_audio(y: np.ndarray, sr: int) -> None:
    """Plays the given audio data."""
    try:
        print("Playing audio...")
        sd.play(y, sr)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"Error playing audio: {e}")


def plot_time(y: np.ndarray, sr: int) -> None:
    """Plot the signal based on time."""
    t = np.linspace(0, sr, len(y))
    plt.figure(figsize=(25, 10))
    plt.plot(t, y, color="r")


if __name__ == "__main__":
    LABELS = []
    RAW_DATA: dict[str, np.ndarray] = {}
    EXTRACTED_DATA: dict[str, list[np.ndarray]] = {}

    # Import the raw data from all audio files in AUDIO_DIR.
    for file in Path.iterdir(AUDIO_DIR):
        if file.suffix == ".wav":
            y, _ = librosa.load(file, sr=DEFAULT_SR, mono=True)
            LABELS.append(file.stem)
            RAW_DATA[file.stem] = y

    # Extract individual keystrokes via onset detection.
    for label, y in RAW_DATA.items():
        keystrokes = isolate_keystrokes_rolling_window(y, sr=DEFAULT_SR)
        EXTRACTED_DATA[label] = keystrokes
        print(f"Found {len([keystrokes])} keystroke(s) for {label}.")

    # Verify each label has 25 samples extracted.
    print()
    for label, keystrokes in EXTRACTED_DATA.items():
        # for y in keystrokes:
        #     plot_time(y, sr=DEFAULT_SR)
        #     plt.show()

        if len(keystrokes) != 25:
            print(f"Error: {len(keystrokes)} detected keystrokes for {label}.")
            sys.exit()
