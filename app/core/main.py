# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.io import wavfile
#
# # Load audio file
# sample_rate, data = wavfile.read("audio/a.wav")
#
# # Create a time axis in seconds
# time = np.linspace(0, len(data) / sample_rate, num=len(data))
#
# # Plot the waveform
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(time, data)
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.title("Audio Waveform")
#
# # Compute the FFT of the audio signal
# fft_coeffs = np.fft.fft(data)
# print(fft_coeffs)
# # Only consider the positive frequencies (first half)
# fft_magnitude = np.abs(fft_coeffs)[: len(fft_coeffs) // 2]
# freqs = np.fft.fftfreq(len(data), d=1 / sample_rate)[: len(fft_coeffs) // 2]
#
# # Plot the FFT magnitude spectrum
# plt.subplot(2, 1, 2)
# plt.plot(freqs, fft_magnitude)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude")
# plt.title("FFT Magnitude Spectrum")
#
# plt.tight_layout()
# plt.show()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile

filename = Path(input("Enter filename: "))

if not filename.exists():
    print("File does not exist!")
    exit()
elif filename.name.endswith(".wav"):
    print("File must be a wav audio file")
    exit()
elif not filename.name.endswith("mono.wav"):
    print("Converting the file to a mono")

# if not filename.
#     sound = AudioSegment.from_wav(filename)
#     sfrom pydub import AudioSegment
# ound = sound.set_channels(1)
#     fm = filename[:-4] + "_mono.wav"
#     sound.export(fm, format="wav")

frequency, data = wavfile.read("audio/a3.wav")
samples = data.shape[0]
duration = samples / frequency

print(f"Frequency: {samples}")
print(f"Duration: {duration}")
print(f"Data: {np.shape(data)}")

t = np.linspace(0, duration, samples)

plt.subplot(2, 1, 1)
plt.plot(t, data[:, 0], "b-")
plt.ylabel("Left")

plt.subplot(2, 1, 2)
plt.plot(t, data[:, 1], "r-")
plt.ylabel("Right")

plt.xlabel("Time (s)")
plt.show()
