import librosa
import numpy as np

from app.core.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


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
