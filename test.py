import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Load audio file
# files = [Path(file) for file in glob.glob("audio/stereo/*.wav")]
files = [Path("audio/stereo/a_norm.wav"), Path("audio/stereo/n_norm.wav")]
blacklist = []
# blacklist = ["a_norm.wav", "n_norm.wav"]

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")

plt.subplot(2, 1, 2)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum")

for file in files:
    if file.name in blacklist:
        continue

    sample_rate, data = wavfile.read(file)
    time = np.linspace(0, len(data) / sample_rate, num=len(data))

    plt.subplot(2, 1, 1)
    plt.plot(time, data, label=file.stem)

    try:
        # Compute the FFT of the audio signal
        fft_coeffs = np.fft.fft(data)
        print(file, fft_coeffs)

        # Only consider the positive frequencies (first half)
        fft_magnitude = np.abs(fft_coeffs)[: len(fft_coeffs) // 2]
        freqs = np.fft.fftfreq(len(data), d=1 / sample_rate)[: len(fft_coeffs) // 2]

        plt.subplot(2, 1, 2)
        plt.plot(freqs, fft_magnitude)
    except Exception:
        continue

plt.legend()
plt.show()
