import logging

import numpy as np

from app.core import constants

log = logging.getLogger(__name__)


class AudioData:
    """Define a unified way of storing audio."""

    def __init__(self, data: np.ndarray, sample_rate: int) -> None:
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
        if data.ndim == 1:
            original_channels = 1
        else:
            original_channels = data.shape[1]

        log.debug(f"Audio sample contains {original_channels} channels")

        if original_channels == target_channels:
            return data

        elif original_channels < target_channels:
            if original_channels == 1:
                log.debug("Duplicating data to all dimensions")
                return np.repeat(data[:, np.newaxis], target_channels, axis=1)
            else:
                log.debug("Tiling data to the rest of dimensions")
                return np.tile(data, (1, target_channels // original_channels))

        else:
            if target_channels == 1:
                log.debug("Averaging data to single dimension")
                return np.mean(data, axis=1)
            else:
                # Otherwise, select the first target_channels channels.
                log.debug("Restricting data to first channels")
                return data[:, :target_channels]

    def _resample_data(self, _: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample the data to the target rate using SciPy's resample.
        Handles both mono and multi-channel data.
        """
        log.error(f"Resampling data from {original_rate} to {target_rate}")
        log.error("This is currently unsupported")
        exit()


class AudioSample:
    """Stores processed labeled audio samples"""

    def __init__(self, processed_audio: AudioData, features: dict):
        self.processed_audio = processed_audio
        self.features = features
