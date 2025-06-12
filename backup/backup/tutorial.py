import sys

import librosa
import librosa.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

mpl.use("TkAgg")

AUDIO_PATH = "audio/scale.wav"
DEFAULT_SR = 44100


def play_audio(audio_data: np.ndarray, sampling_rate: int):
    """Plays the given audio data."""
    try:
        print("Playing audio...")
        sd.play(audio_data, sampling_rate)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"Error playing audio: {e}")


def record_audio(duration_seconds: int, sampling_rate: int):
    """Records audio from the default microphone for a given duration."""
    try:
        print(f"Recording for {duration_seconds} seconds...")
        recording = sd.rec(
            int(duration_seconds * sampling_rate),
            samplerate=sampling_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("Recording finished.")
        return recording.flatten(), sampling_rate
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None, None


y = None
sr = None

try:
    print(f"Attempting to load audio from: {AUDIO_PATH}")
    y, sr = librosa.load(AUDIO_PATH, sr=None)
    print(f"Audio loaded successfully. Sampling rate: {sr} Hz, Duration: {len(y) / sr:.2f} seconds")

except Exception as e:
    print(f"\nError loading audio file: {e}")
    print("The specified audio file could not be loaded.")

    while True:
        try:
            duration_str = input("Please enter a duration (in seconds) to record audio from your microphone: ")
            duration_seconds = float(duration_str)
            if duration_seconds <= 0:
                print("Duration must be a positive number.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 5.5).")

    print(f"\nProceeding to record {duration_seconds}s of audio.")
    y, sr = record_audio(duration_seconds, DEFAULT_SR)

    if y is not None:
        print("Audio recorded successfully.")
    else:
        print("Failed to record audio. Exiting.")
        sys.exit()


def plot_spectogram(signal: np.ndarray, name: str):
    """Compute power spectogram with Short-Time Fourier Transform and plot result."""
    spectogram = librosa.amplitude_to_db(librosa.stft(signal))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequecy pwoer spectogram for {name}")
    plt.xlabel("Time")
    plt.show()


def plot_time(signal: np.ndarray, name: str):
    """Plot the signal in the time domain."""
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(f"Time Domain Signal: {name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def amplitude_envolope(signal: np.ndarray, frame_size: int, hop_length: int):
    """Calculate the amplitude envelope of the signal from scratch."""
    result = []

    for i in range(0, len(signal), hop_length):
        current_frame_ae = max(signal[i : i + frame_size])
        result.append(current_frame_ae)

    return np.array(result)


def rms(signal: np.ndarray, frame_size: int, hop_length: int):
    """Calculate the Root-Mean Square of the signal from scratch."""
    result = []
    for i in range(0, len(signal), hop_length):
        rms_current_frame = np.sqrt((np.sum(signal[i : i + frame_size] ** 2)) / frame_size)
        result.append(rms_current_frame)

    return np.array(result)


def zcr(signal: np.ndarray, frame_size: int, hop_length: int):
    """Calculate the Root-Mean Square of the signal from scratch."""
    result = []
    for i in range(0, len(signal), hop_length):
        zcr_current_frame = np.sum(signal[i : i + frame_size]) / 2
        result.append(zcr_current_frame)

    return np.array(result)


def calculate_band_energy_ratio(spectogram: np.ndarray, split_frequency: int, sample_rate: int) -> int:
    """Calculate band energy ratio from scratch."""
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / spectogram.shape[0]
    split_frequency_bin = int(np.floor(split_frequency / frequency_delta_per_bin))

    power_spec = np.abs(spectogram) ** 2
    power_spec = power_spec.T

    band_energy_ratio = []
    for frequencies_in_frame in power_spec:
        sum_power_low_frequencies = np.sum(frequencies_in_frame[:split_frequency_bin])
        sum_power_high_frequencies = np.sum(frequencies_in_frame[split_frequency_bin:])
        ber_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(ber_current_frame)

    return np.array(band_energy_ratio)


# plot_time(y, "Broken out in the wild")
# sf.write(AUDIO_PATH, y, sr)

# FRAME_SIZE = 1024
# HOP_LENGTH = 512
#
# sc_y = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
# sb_y = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
# frames = range(len(sc_y))
# t = librosa.frames_to_time(frames)
#
# plt.figure(figsize=(25, 10))
# plt.plot(t, sc_y, color="b")
# plt.plot(t, sb_y, color="r")
# plt.show()

# filter_banks = librosa.filters.mel(n_fft=FRAME_SIZE, sr=22050, n_mels=10)
# print(filter_banks.shape)
#
# plt.figure(figsize=(25, 10))
# librosa.display.specshow(filter_banks, sr=sr, x_axis="linear")
# plt.show()

# s_y = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
# ber_y = calculate_band_energy_ratio(s_y, 2000, sr)
#
# frames = range(len(ber_y))
# t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
#
# plt.figure(figsize=(25, 10))
# plt.plot(t, ber_y)
# plt.show()

# mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=90)
# log_mel_spectogram = librosa.power_to_db(mel_spectogram)
#
# plt.figure(figsize=(25, 10))
# librosa.display.specshow(log_mel_spectogram, x_axis="time", y_axis="mel", sr=sr)
# plt.colorbar(format="%+2.f")
# plt.show()

# sample_duration = 1 / s_y.size
# audio_duration = sample_duration * len(y)

# print(y.shape)
# y_fft = np.fft.fft(y)
# magnitude = np.abs(y_fft)
# phase = np.angle(y_fft)
# frequency = np.linspace(0, sr, len(magnitude))
#
# print(y_fft[0], magnitude[0])

# s_y = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
# y_y = np.abs(s_y) ** 2
# y_log_scale = librosa.power_to_db(y_y)
# print(y_y.shape)
#
# plt.figure(figsize=(25, 10))
# librosa.display.specshow(y_log_scale, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="log")
# plt.colorbar(format="%+2.f")
# plt.show()

# plt.figure(figsize=(18, 8))
# plt.plot(frequency, magnitude, color="r")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")

# ae_y = amplitude_envolope(y, FRAME_SIZE, HOP_LENGTH)
# rms_y = rms(y, FRAME_SIZE, HOP_LENGTH)
# rms_y = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
# zcr_y = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]


# print(ae_y.shape)
# print(rms_y.shape)
# print(zcr_y.shape)

# print(1 / 523)
#
# ft = sp.fft.fft(y)
# magnitude = np.abs(ft)
# frequency = np.linspace(0, sr, len(magnitude))
#
# samples = range(len(y))
# t = librosa.samples_to_time(samples, sr=sr)
#
# f = 523
# phase = 0.55
# sin = 0.1 * np.sin(2 * np.pi * (f * t - phase))
#
# plt.figure(figsize=(18, 8))
# plt.plot(t[10000:10400], y[10000:10400])
# plt.plot(t[10000:10400], sin[10000:10400], color="r")
# plt.fill_between(t[10000:10400], sin[10000:10400] * y[10000:10400], color="y")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")

# librosa.display.waveshow(y, alpha=0.5)
# plt.plot(t, ae_y, color="r")
# plt.plot(t, rms_y, color="g")
# plt.plot(t, zcr_y, color="y")
# plt.title("y")
