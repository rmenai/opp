import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Load audio file
# files = [Path(file) for file in glob.glob("audio/stereo/*.wav")]
files = [
    Path("audio/mono/a.wav"),
    Path("audio/mono/a1.wav"),
    Path("audio/mono/a2.wav"),
    Path("audio/mono/a3.wav"),
    Path("audio/mono/a4.wav"),
    Path("audio/mono/a5.wav"),
    Path("audio/mono/a6.wav"),
    Path("audio/mono/a7.wav"),
    Path("audio/mono/n4_.wav"),
    Path("audio/mono/n5_.wav"),
    Path("audio/mono/n6_.wav"),
    Path("audio/mono/n7_.wav"),
    Path("audio/mono/n1.wav"),
    Path("audio/mono/n2.wav"),
    Path("audio/mono/n3.wav"),
    Path("audio/mono/n3.wav"),
]


def calculate_average(files):
    average = {}

    for file in files:
        sample_rate, data = wavfile.read(file)

        fft_coeffs = np.fft.fft(data)
        fft_magnitude = np.abs(fft_coeffs)[: len(fft_coeffs) // 2]
        freqs = np.fft.fftfreq(len(data), d=1 / sample_rate)[: len(fft_coeffs) // 2]

        for k in range(len(freqs)):
            if not average.get(freqs[k], None):
                average[freqs[k]] = 0

            average[freqs[k]] += fft_magnitude[k]

    freqs = np.array(list(average.keys()))
    magnitudes = np.array(list(average.values())) / len(files)

    return freqs, magnitudes


freqs_a, magnitudes_a = calculate_average(files[:8])
freqs_n, magnitudes_n = calculate_average(files[8:])

### AUDIO SAMPLE TO DETECT
file = Path("audio/mono/a1")

sample_rate, data = wavfile.read(file)

fft_coeffs = np.fft.fft(data)
fft_magnitude = np.abs(fft_coeffs)[: len(fft_coeffs) // 2]
freqs = np.fft.fftfreq(len(data), d=1 / sample_rate)[: len(fft_coeffs) // 2]

# for k in range(len(freqs)):
#     if not average.get(freqs[k], None):
#         average[freqs[k]] = 0
#
#     average[freqs[k]] += fft_magnitude[k]

# plt.figure(figsize=(12, 6))
#
# plt.title("FFT Magnitude Spectrum")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")
#
# plt.plot(freqs_a, magnitudes_a, color="red")
# plt.plot(freqs_n, magnitudes_n, color="blue")
#
# plt.legend()
# plt.show()
