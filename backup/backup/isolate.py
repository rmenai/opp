import librosa
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SR = 44100
KEYSTROKE_LENGTH = 14400
FRAME_SIZE = 2048
HOP_LENGTH = 512


def isolate_keystrokes_rolling_window(
    y: np.ndarray,
    sr: int,
    keystroke_len: int = KEYSTROKE_LENGTH,
    # --- These are the NEW, intuitive tuning knobs ---
    trigger_level: float = 4.0,
    release_level: float = 1.5,
    cooldown_ms: int = 100,
) -> List[np.ndarray]:
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
    hop_length = 512
    rms = librosa.feature.rms(y=y_filtered, frame_length=1024, hop_length=hop_length)[0]

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


def isolate_keystrokes_manual(
    y: np.ndarray,
    sr: int,
    n_fft: int = FRAME_SIZE,
    hop_length: int = HOP_LENGTH,
    threshold_multiplier: float = 2,
    keystroke_len: int = KEYSTROKE_LENGTH,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """
    Isolates individual keystrokes from a raw audio signal.

    Args:
        y: The input audio time series.
        sr: The sampling rate of y.
        n_fft: The FFT window size.
        hop_length: The hop length for the STFT.
        threshold_multiplier: How many standard deviations above the mean to set the threshold.
                              Tune this value based on your recording's noise level.
        keystroke_len: The fixed number of samples for each extracted keystroke.

    Returns:
        A tuple containing:
        - A list of numpy arrays, where each array is an isolated keystroke.
        - The calculated energy curve.
        - The calculated energy threshold.
    """
    # 1. Perform STFT and get the magnitude (which relates to energy)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 2. Sum coefficients across frequencies to get 'energy' per time frame
    # We use the magnitude from the STFT for this
    energy = np.sum(np.abs(D), axis=0)

    # 3. Define an energy threshold
    # A dynamic threshold is more robust than a fixed one
    threshold = np.mean(energy) + threshold_multiplier * np.std(energy)

    # 4. Find where the energy crosses the threshold (marking keystroke onsets)
    # We look for frames where the energy is above the threshold AND the previous frame was below.
    onsets = []
    is_over_threshold = energy > threshold
    for i in range(1, len(is_over_threshold)):
        if is_over_threshold[i] and not is_over_threshold[i - 1]:
            # Convert frame index to sample index
            start_sample = librosa.frames_to_samples(i, hop_length=hop_length)
            onsets.append(start_sample)

    # 5. Extract fixed-length keystrokes
    extracted_keys = []
    for start_sample in onsets:
        end_sample = start_sample + keystroke_len
        # Ensure we don't go past the end of the audio file
        if end_sample < len(y):
            keystroke = y[start_sample:end_sample]
            extracted_keys.append(keystroke)

    return extracted_keys, energy, threshold


def isolate_keystrokes_librosa(
    y: np.ndarray,
    sr: int,
    keystroke_len: int = KEYSTROKE_LENGTH,
) -> list[np.ndarray]:
    """
    Isolates individual keystrokes using librosa's built-in onset detector for reliability.

    Args:
        y: The input audio time series.
        sr: The sampling rate of y.
        keystroke_len: The fixed number of samples for each extracted keystroke.

    Returns:
        A list of numpy arrays, where each array is an isolated keystroke.
    """
    # 1. Use librosa's onset detection function.
    #    - 'units="samples"' gives us the start time in samples, which is perfect for slicing.
    #    - 'backtrack=True' finds the local minimum of energy before the peak for a more precise start.
    #    - You can adjust 'pre_avg', 'post_avg', 'pre_max', 'post_max', and 'delta' for peak picking.
    onset_samples = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="samples",
        backtrack=True,
    )

    # 2. Extract fixed-length keystrokes based on the detected onsets
    extracted_keys = []
    for start_sample in onset_samples:
        end_sample = start_sample + keystroke_len
        # Ensure we don't go past the end of the audio file
        if end_sample < len(y):
            keystroke = y[start_sample:end_sample]
            extracted_keys.append(keystroke)

    return extracted_keys


def isolate_keystrokes_tuned(
    y: np.ndarray,
    sr: int,
    keystroke_len: int = KEYSTROKE_LENGTH,
) -> list[np.ndarray]:
    """
    Isolates keystrokes using a TUNED librosa onset detector.

    Args:
        y: The input audio time series.
        sr: The sampling rate of y.
        keystroke_len: The fixed number of samples for each extracted keystroke.

    Returns:
        A list of numpy arrays, where each array is an isolated keystroke.
    """
    # --- 1. PRE-PROCESS THE AUDIO ---
    # Apply a high-pass filter to remove low-frequency hum and rumble.
    # This is a HUGE help for isolating sharp clicks.
    y_filtered = highpass_filter(y, sr, cutoff=500)  # Cut off frequencies below 500 Hz

    # --- 2. USE THE TUNED ONSET DETECTOR ---
    # We are adding 'wait' and 'delta' to make it less sensitive.
    hop_length = 512

    # Calculate how many frames to wait (e.g., 50ms)
    # 50ms = 0.050 seconds
    # wait_seconds = 0.050
    # wait_frames = int(librosa.time_to_frames(wait_seconds, sr=sr, hop_length=hop_length))

    onset_samples = librosa.onset.onset_detect(
        y=y_filtered,  # Use the filtered audio!
        sr=sr,
        hop_length=hop_length,
        units="samples",
        backtrack=True,
        # --- KEY PARAMETERS TO TUNE ---
        pre_avg=int(librosa.time_to_frames(0.02, sr=sr, hop_length=hop_length)),  # Average over 20ms before
        post_avg=int(librosa.time_to_frames(0.02, sr=sr, hop_length=hop_length)),  # Average over 20ms after
        pre_max=int(
            librosa.time_to_frames(0.02, sr=sr, hop_length=hop_length),
        ),  # A peak must be the max in a 20ms window before it
        post_max=int(librosa.time_to_frames(0.02, sr=sr, hop_length=hop_length)),  # and a 20ms window after it
        delta=0.2,  # How much higher a peak must be than the local average. Increase this to reject noise.
        wait=int(
            librosa.time_to_frames(0.05, sr=sr, hop_length=hop_length),
        ),  # Wait 50ms between detections to avoid double-triggers.
    )

    # (The rest of the function is the same)
    extracted_keys = []
    for start_sample in onset_samples:
        end_sample = start_sample + keystroke_len
        if end_sample < len(y):
            # IMPORTANT: Extract from the ORIGINAL audio (y), not the filtered one (y_filtered)
            # so you keep the full sound of the keystroke.
            keystroke = y[start_sample:end_sample]
            extracted_keys.append(keystroke)

    return extracted_keys


def plot_isolation_process(y: np.ndarray, sr: int, energy: np.ndarray, threshold: float):
    """
    Visualizes the isolation process, similar to Figure 1 in the paper.
    """
    fig, ax1 = plt.subplots(figsize=(25, 10))

    # Plot the raw audio signal
    t_signal = np.linspace(0, len(y) / sr, num=len(y))
    ax1.plot(t_signal, y, color="b", label="Raw Audio Signal", alpha=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Create a second y-axis for the energy curve
    ax2 = ax1.twinx()
    # Calculate the time axis for the energy frames
    t_energy = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=512)
    ax2.plot(t_energy, energy, color="r", label="Calculated Energy")
    ax2.axhline(y=threshold, color="g", linestyle="--", label="Energy Threshold")
    ax2.set_ylabel("Energy", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.tight_layout()
    plt.title("Keystroke Isolation Process")
    plt.legend()
    plt.show()


def visualize_isolation_process(y: np.ndarray, sr: int, label: str = "Unkown"):
    """Visualize and play an example."""
    _, energy_curve, energy_threshold = isolate_keystrokes_manual(y, sr)
    print(f"\nShowing isolation plot for key '{label}'...")
    plot_isolation_process(y, sr, energy_curve, energy_threshold)

    # y_filtered = highpass_filter(RAW_DATA["q"], DEFAULT_SR, cutoff=500)
    # plot_time(y_filtered, sr=DEFAULT_SR)
    # plt.show()
    # print("\n--- Verification ---")
    # print("Final EXTRACTED_DATA contains the following keys:")
    # for label, strokes in EXTRACTED_DATA.items():
    #     print(f"  - '{label}': A list containing {len(strokes)} extracted keystroke(s).")
