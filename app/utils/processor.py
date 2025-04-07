import numpy as np
from core.audio import AudioData


class AudioProcessor:
    """Processes, cuts and normalizes audio"""

    def normalize(self, audio: AudioData) -> AudioData:
        """
        Resample the data to the target rate using SciPy's resample.
        Handles both mono and multi-channel data.
        """
        data = audio.data / np.max(np.abs(audio.data))
        return AudioData(data, audio.sample_rate)

    def compute_fft(self, audio_data: AudioData) -> np.ndarray:
        return np.abs(np.fft.fft(audio_data.data))
