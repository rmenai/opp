import logging

import sounddevice as sd
from core.audio import AudioData
from core.constants import constants
from scipy.io import wavfile

log = logging.getLogger(__name__)


class AudioRecorder:
    """Captures audio, handles io."""

    def from_file(self, filepath: str):
        """
        Load audio from a WAV file using SciPy.
        """
        sample_rate, data = wavfile.read(filepath)
        log.info(f"Imported {filepath}")
        return AudioData(data, sample_rate)

    def record_live(self, duration: float):
        """
        Record live audio for the specified duration using sounddevice.
        The recorded audio will have the target SAMPLE_RATE and CHANNELS.
        """
        log.info(f"Starting audio recording")

        data = sd.rec(
            int(duration * constants.SAMPLE_RATE),
            samplerate=constants.SAMPLE_RATE,
            channels=constants.CHANNELS,
        )

        sd.wait()
        log.info(f"Finished recording {duration} seconds of audio")

        return AudioData(data, constants.SAMPLE_RATE)
