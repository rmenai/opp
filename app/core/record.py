"""
This script records audio and logs keyboard events from a Linux input device (e.g., /dev/input/event0).
When any key is pressed, audio recording starts and all subsequent key events are logged with timestamps.
Pressing ESC stops the recording and saves both the audio (as a WAV file) and the keyboard events (as a text file).
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from evdev import InputDevice, ecodes

from app.core.config import AUDIO_DIR, CHANNELS, DTYPE_RECORD, OUTPUT_FILENAME, RECORD_SAMPLE_RATE, TIME_SKIP

dev = InputDevice("/dev/input/event0")


class AudioRecorder:
    def __init__(self, filename: str, time_skip: int) -> None:
        self.filename = filename
        self._recording = False
        self.frames = []
        self.start_time = None
        self.time_skip = time_skip

    def _record_audio(self) -> None:
        """Continuously record audio chunks while the recording flag is set."""
        with sd.InputStream(
            samplerate=RECORD_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE_RECORD,
        ) as stream:
            self.start_time = time.time()
            while self._recording:
                audio_chunk, overflowed = stream.read(1024)
                if not overflowed:
                    self.frames.append(audio_chunk)

    def start(self) -> None:
        """Start the audio recording in a separate thread."""
        self._recording = True
        self.thread = threading.Thread(target=self._record_audio)
        self.thread.start()
        print("Audio recording started...")

    def stop(self) -> np.ndarray:
        """Stop the audio recording and save to a WAV file."""
        if self._recording:
            self._recording = False
            self.thread.join()
            print("Audio recording stopped.")

        return self.save()

    def save(self) -> np.ndarray:
        """Save the recorded audio frames to a WAV file."""
        if not self.frames:
            print("No audio frames to save.")
            return np.array([])

        if self.time_skip:
            samples_to_skip = int(RECORD_SAMPLE_RATE * self.time_skip)
            recording = np.concatenate(self.frames, axis=0)
            recording = recording[samples_to_skip:-samples_to_skip]
            self.start_time += self.time_skip  # Don't forget to update the timestamp.
        else:
            recording = np.concatenate(self.frames, axis=0)

        if self.filename:
            sf.write(self.filename, recording, RECORD_SAMPLE_RATE)
            print(f"Audio saved to {self.filename}")

        return recording.squeeze(), RECORD_SAMPLE_RATE


async def main():
    """Main function to capture keyboard events and trigger audio recording."""
    try:
        dev = InputDevice("/dev/input/event0")
        print(f"Listening to keyboard: {dev.name}")
    except PermissionError:
        print("Permission denied. You might need to run this script with sudo.")
        return
    except FileNotFoundError:
        print(f"Input device not found. Is {dev} correct?")
        return

    raw_dir = AUDIO_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    recorder = AudioRecorder(raw_dir / f"{OUTPUT_FILENAME}.wav", TIME_SKIP)
    keyboard_events = []

    print("Press any key to start recording audio and logging keystrokes...")
    print("Press 'ESC' to stop.")

    is_recording = False

    async for ev in dev.async_read_loop():
        if not is_recording:
            is_recording = True
            recorder.start()

        if ev.type == ecodes.EV_KEY:  # Is not <ESC>.
            if ev.code == ecodes.KEY_ESC and ev.value == 1:
                break

            event_time = time.time()
            key_event = {"timestamp": event_time, "code": ev.code, "value": ev.value}  # 0: up, 1: down, 2: hold()
            keyboard_events.append(key_event)
            print(repr(ev))

    recorder.stop()

    data = {
        "sample_rate": RECORD_SAMPLE_RATE,
        "channels": CHANNELS,
        "dtype": DTYPE_RECORD,
        "start_time": recorder.start_time,
        "num_keystrokes": int(len(keyboard_events[2:]) / 2),
        "keystrokes": keyboard_events[2:],  # First two keys are for activation
    }

    with Path.open(raw_dir / f"{OUTPUT_FILENAME}.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Keyboard events saved to {OUTPUT_FILENAME}.json file")
