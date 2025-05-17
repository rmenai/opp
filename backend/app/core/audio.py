"""Old file used to define audio."""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

from app.core import constants

log = logging.getLogger(__name__)


class AudioData:
    """Define a unified way of storing audio."""

    def __init__(self, data: np.ndarray, sample_rate: int) -> None:
        """Initialize audio data fields."""
        data = self._unify_channels(data, target_channels=constants.CHANNELS)

        if sample_rate != constants.SAMPLE_RATE:
            data = self._resample_data(data, sample_rate, constants.SAMPLE_RATE)
            sample_rate = constants.SAMPLE_RATE

        self.data = data
        self.sample_rate = sample_rate

    def _unify_channels(self, data: np.ndarray, target_channels: int) -> np.ndarray:
        """
        Convert data to have the desired number of channels.

        - If the input is mono but target is stereo, duplicate the channel.
        - If input has more channels than target, either average or select the first ones.
        """
        original_channels = 1 if data.ndim == 1 else data.shape[1]

        log.debug("Audio sample contains %s channels", original_channels)

        if original_channels == target_channels:
            return data

        if original_channels < target_channels:
            if original_channels == 1:
                log.debug("Duplicating data to all dimensions")
                return np.repeat(data[:, np.newaxis], target_channels, axis=1)
            log.debug("Tiling data to the rest of dimensions")
            return np.tile(data, (1, target_channels // original_channels))

        if target_channels == 1:
            log.debug("Averaging data to single dimension")
            return np.mean(data, axis=1)
        # Otherwise, select the first target_channels channels.
        log.debug("Restricting data to first channels")
        return data[:, :target_channels]

    def _resample_data(self, _: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample the data to the target rate using SciPy's resample.

        Handles both mono and multi-channel data.
        """
        log.error("Resampling data from %s to %s", original_rate, target_rate)
        log.error("This is currently unsupported")
        sys.exit()


class AudioSample:
    """Stores processed labeled audio samples."""

    def __init__(self, audio: AudioData, fft: np.ndarray, label: str) -> None:
        """Initialize fields."""
        self.audio = audio
        self.fft = fft
        self.label = label

    def visualize(self) -> None:
        """Plot."""
        plt.figure(figsize=(10, 4))

        if self.audio.data.ndim == 1:
            plt.plot(self.audio.data, label="Mono")
        else:
            for i in range(self.audio.data.shape[1]):
                plt.plot(self.audio.data[:, i], label=f"Channel {i + 1}")

        plt.title(f"Audio Waveform ({self.label})")
        plt.xlabel("Data Points")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()
