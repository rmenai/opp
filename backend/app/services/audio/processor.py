"""Functions to process audio."""

import logging

import numpy as np
from scipy.signal import find_peaks

from app.core import constants
from app.core.audio import AudioData, AudioSample

log = logging.getLogger(__name__)


class AudioProcessor:
    """Process, cut and normalize audio."""

    def average(self, samples: list[AudioSample]) -> AudioSample:
        """Compute the average of multiple audio samples."""
        sr = samples[0].audio.sample_rate

        lengths = [s.audio.data.shape[0] for s in samples]
        min_len = min(lengths)

        stacked_data = []
        for s in samples:
            d = s.audio.data[:min_len]
            if d.ndim == 1:
                d = d[:, None]
            stacked_data.append(d.astype(np.float64))

        data_stack = np.stack(stacked_data, axis=0)
        mean_data = data_stack.mean(axis=0)
        mean_data = mean_data.astype(samples[0].audio.data.dtype)

        fft_stack = np.stack([s.fft[:min_len].astype(np.float64) for s in samples], axis=0)
        mean_fft = fft_stack.mean(axis=0)

        avg_label = "avg(" + ",".join(s.label for s in samples) + ")"

        return AudioSample(audio=AudioData(mean_data, sr), fft=mean_fft, label=avg_label)

    def compute_fft(self, audio: AudioData) -> np.ndarray:
        """Compute the fft."""
        return np.abs(np.fft.fft(audio.data))

    def find_peaks(self, audio: AudioData) -> list[int]:
        """Automatically find the peaks in the audio."""
        data = audio.data

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        peaks, _ = find_peaks(data, prominence=constants.PROMINENCE, distance=audio.sample_rate * constants.DISTANCE)
        log.info("Peaks: %s", peaks)

        if not peaks.any():
            log.error("No peaks found in signal")

        return peaks.tolist()

        return find_peaks(audio.data, prominence=1)

    def resize(self, audio: AudioData, time: int) -> AudioData:
        """Trim the audio data to a fixed length, either trimming or padding."""
        center_sample = time
        target_samples = constants.LENGTH
        total_samples, n_channels = audio.data.shape

        half = target_samples // 2
        start = center_sample - half
        end = center_sample + (target_samples - half)

        if total_samples < target_samples:
            output = np.zeros((target_samples, n_channels), dtype=audio.data.dtype)

            pad_start = max(0, -start)
            data_start = max(0, start)
            data_end = min(total_samples, end)

            if data_end > data_start:
                pad_start = max(0, -start)
                data_start = max(0, start)
                data_end = min(total_samples, end)
                pad_end = pad_start + (data_end - data_start)

                output[pad_start:pad_end] = audio.data[data_start:data_end]

        else:
            start = max(0, start)
            end = min(total_samples, end)

            if end - start < target_samples:
                if start == 0:
                    end = target_samples
                elif end == total_samples:
                    start = total_samples - target_samples

            output = audio.data[start:end]

        return AudioData(output, audio.sample_rate)

    def normalize(self, audio: AudioData) -> AudioData:
        """
        Resample the data to the target rate using SciPy's resample.

        Handles both mono and multi-channel data.
        """
        data = audio.data / np.max(np.abs(audio.data))
        return AudioData(data, audio.sample_rate)

    def process(self, audio: AudioData, peaks: list[int] | None = None, label: str = "") -> list[AudioSample]:
        """Process, cut and normalize audio."""
        if peaks is None:
            peaks = []

        audio = self.normalize(audio)

        if not peaks:
            peaks = self.find_peaks(audio)

        samples = []
        for peak in peaks:
            audio_part = self.resize(audio, peak)
            fft_part = self.compute_fft(audio_part)

            samples.append(AudioSample(audio_part, fft_part, f"Processed ({label}) ({peak})"))

        return samples
