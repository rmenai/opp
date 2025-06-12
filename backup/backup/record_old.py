import os
import sys
import time

import matplotlib as mpl
import numpy as np
import sounddevice as sd

mpl.use("TkAgg")

SAMPLE_RATE = 44100
NUM_PRESSES_PER_KEY = 25

KEYS_TO_RECORD = [
    "q",
    "w",
    "e",
    "r",
    "t",
    "y",
    "u",
    "i",
    "o",
    "p",
    "a",
    "s",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "z",
    "x",
    "c",
    "v",
    "b",
    "n",
    "m",
    "space",
]


def record_continuous_audio(sample_rate: int):
    """
    Starts a recording and waits for the user to press Enter to stop it.
    Returns the recorded audio data as a NumPy array.
    """
    input("   Press Enter to START recording for this key...")
    print("   RECORDING STARTED. Press the key multiple times.")
    print(f"   Perform your {NUM_PRESSES_PER_KEY} presses now, varying pressure...")

    recorded_frames = []

    def callback(indata, frames, time, status) -> None:  # noqa: ANN001, ARG001
        if status:
            print(status, file=sys.stderr)
        recorded_frames.append(indata.copy())

    # Start stream recording
    # Using a high blocksize might introduce latency at the start/end but is simpler
    # than managing smaller blocks if we don't need real-time processing during rec.
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback)
    with stream:
        input(
            f"   Press Enter again to STOP recording for this key after you've made ~{NUM_PRESSES_PER_KEY} presses...",
        )
        # Stream is automatically stopped when exiting the 'with' block after Enter is pressed

    print("   RECORDING STOPPED.")
    if not recorded_frames:
        print("   Warning: No audio data was recorded.")
        return np.array([], dtype=np.float32)

    recording = np.concatenate(recorded_frames, axis=0)
    return recording.flatten()


def collect_keystroke_data_continuous():
    """Manages the continuous data collection process for each key."""
    print("Starting keystroke data collection.")
    print("For each key, you will be prompted to start a recording.")
    print(f"During that recording, please press the specified key {NUM_PRESSES_PER_KEY} times.")
    print("Remember to VARY THE PRESSURE of your key presses (light, medium, hard).")
    print("Ensure you are in a quiet environment.")
    print("-" * 30)

    all_recordings_per_key = {}  # Dictionary to store recordings: {'key_name': single_long_recording}
    initial_input = input(
        "Press Enter to begin the collection process for the first key, or type 'skip' to review keys first: ",
    )
    if initial_input.lower() == "skip":
        print("\nReviewing keys to be recorded:")
        for key_name in KEYS_TO_RECORD:
            print(f"- {key_name}")
        input("\nPress Enter to begin the collection process...")

    for key_index, key_name in enumerate(KEYS_TO_RECORD):
        print(f"\n--- Preparing for key: '{key_name.upper()}' ({key_index + 1}/{len(KEYS_TO_RECORD)}) ---")
        print(f"   You will need to press this key approx. {NUM_PRESSES_PER_KEY} times in one recording.")
        print("   Try to make distinct presses with short pauses in between.")

        audio_data = record_continuous_audio(SAMPLE_RATE)

        if audio_data.size > 0:
            all_recordings_per_key[key_name] = audio_data
            print(f"   Recording for '{key_name.upper()}' captured ({audio_data.shape[0] / SAMPLE_RATE:.2f} seconds).")

            # Optional: Display a spectrogram of the recorded audio for quick check
            # if plt is not None:
            #     plt.figure(figsize=(10, 4))
            #     D = librosa.stft(audio_data)
            #     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            #     librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis='time', y_axis='hz')
            #     plt.colorbar(format='%+2.0f dB')
            #     plt.title(f"Spectrogram for all presses of '{key_name}'")
            #     plt.tight_layout()
            #     plt.show(block=False)
            #     plt.pause(2) # Show plot for 2 seconds
            #     plt.close()
        else:
            print(f"   Skipping key '{key_name.upper()}' due to empty recording.")

        time.sleep(0.5)  # Brief pause before next key prompt

    print("\n--- Data Collection Complete! ---")
    print(f"Collected recordings for {len(all_recordings_per_key)} keys.")
    return all_recordings_per_key


def save_continuous_recordings(recordings_dict, base_path="keystroke_data_continuous"):
    """Saves the recorded audio data to files (one file per key)."""

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    print(f"\nSaving recordings to directory: '{os.path.abspath(base_path)}'")

    for key_name, audio_data in recordings_dict.items():
        # Sanitize filename: replace spaces, slashes, etc.
        safe_key_name = key_name.replace(" ", "_").replace("/", "_slash_").replace("\\", "_backslash_")
        safe_key_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in safe_key_name)

        filename_wav = os.path.join(base_path, f"{safe_key_name}.wav")
        filename_npy = os.path.join(base_path, f"key_{safe_key_name}.npy")  # Fallback

        try:
            import soundfile as sf

            sf.write(filename_wav, audio_data, SAMPLE_RATE)
            print(f"Saved: {filename_wav}")
        except ImportError:
            print("Soundfile library not found. Saving as .npy instead.")
            print("Consider installing it: pip install soundfile")
            np.save(filename_npy, audio_data)
            print(f"Saved: {filename_npy}")
        except Exception as e:
            print(f"Error saving audio for key '{key_name}' as {filename_wav} or {filename_npy}: {e}")


if __name__ == "__main__":
    print("Available audio input devices:")
    try:
        print(sd.query_devices())
        default_device_info = sd.query_devices(kind="input")
        if default_device_info and isinstance(default_device_info, dict) and "name" in default_device_info:
            print(f"Using default input device: {default_device_info['name']}")
        else:
            print("Could not determine default input device name. Using system default.")
    except Exception as e:
        print(f"Could not query audio devices: {e}")
        print("Please ensure you have a microphone connected and configured.")
        sys.exit(1)

    # --- Start Data Collection ---
    collected_data = collect_keystroke_data_continuous()

    # --- Save the collected data ---
    if collected_data:
        save_choice = input("\nDo you want to save the collected recordings? (yes/no): ").strip().lower()
        if save_choice == "yes":
            output_directory = input(
                "Enter the base directory name to save recordings (e.g., 'my_continuous_keystrokes'): ",
            )
            if not output_directory.strip():  # Check if empty or only whitespace
                output_directory = "keystroke_audio_data_continuous"
            save_continuous_recordings(collected_data, base_path=output_directory)
        else:
            print("Recordings will not be saved.")
    else:
        print("No data was collected.")

    print("\nProgram finished.")
