import logging

import numpy as np

from app.core import settings
from app.core.audio import AudioData, AudioSample
from app.core.processor import AudioProcessor
from app.utils.recorder import AudioRecorder

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_sample(label):
    duration = float(input("enter duration: "))

    audio = recorder.record_live(duration)
    parts = processor.process(audio, label=label)

    sample = AudioSample(audio, np.ndarray([]), "")

    for part in parts:
        part.visualize()

    sample.visualize()

    return parts


if __name__ == "__main__":
    log.info("Project successful")

    recorder = AudioRecorder()
    processor = AudioProcessor()

    # --- load and average your reference samples ---
    a = get_sample("a")
    b = get_sample("b")

    average_a = processor.average(a)
    average_b = processor.average(b)

    # --- compute reference FFTs ---
    fft_avg_a = processor.compute_fft(average_a.audio)
    fft_avg_b = processor.compute_fft(average_b.audio)

    log.info("Reference FFTs computed. Entering live classification loop...")

    while True:
        user_in = input("\nEnter duration in seconds (blank to exit): ")
        if not user_in.strip():
            log.info("Exiting.")
            break

        try:
            duration = float(user_in)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if duration <= 0:
            log.info("Exiting.")
            break

        # record live audio
        audio = recorder.record_live(duration)

        # split into parts (e.g. individual keystroke events, syllables, etc.)
        parts = processor.process(audio)

        for idx, part in enumerate(parts):
            # compute FFT of this segment
            fft_part = processor.compute_fft(part.audio)

            # measure Euclidean distance to each reference FFT
            dist_a = np.linalg.norm(fft_part - fft_avg_a)
            dist_b = np.linalg.norm(fft_part - fft_avg_b)

            # choose the closest
            label = "a" if dist_a < dist_b else "b"

            print(f"[part {idx}] → label={label} (dist_a={dist_a:.2f}, dist_b={dist_b:.2f})")

            # # optionally visualize each segment
            # part.visualize()

# import logging
#
# import numpy as np
#
#
# log = logging.getlogger(__name__)
#
#
#
#
# if __name__ == "__main__":
#     log.info("project successful")
#
#     recorder = audiorecorder()
#     processor = audioprocessor()
#
#     samples = []
#
#     # while true:
#     #     label = input("enter label: ")
#     #
#     #     if not label:
#     #        break
#     #
#     #     duration = float(input("enter duration: "))
#     #
#     #     audio = recorder.record_live(duration)
#     #     parts = processor.process(audio, label=label)
#     #
#     #     sample = audiosample(audio, np.ndarray([]), "")
#     #
#     #     for part in parts:
#     #         part.visualize()
#     #
#     #     sample.visualize()
#     #
#     #     samples += parts
#
#     a = get_sample("a")
#     b = get_sample("b")
#
#     average_a = processor.average(a)
#     average_b = processor.average(b)
#
#     # for sample in samples:
#     #     sample.visualize()
#
#
#
#     # filenames = [
#     #     "a6.wav",
#     #     "a1.wav",
#     #     "a2.wav",
#     #     "a3.wav",
#     # ]
#     #
#     # samples = []
#     #
#     # for filename in filenames:
#     #     audio = recorder.from_file(settings.audio_dir / filename)
#     #     # sample = audiosample(audio, np.ndarray([]), "")
#     #     # sample.visualize()
#     #     parts = processor.process(audio, label=filename)
#     #     samples += parts
#     #
#     # average_sample = processor.average(samples)
#     # average_sample.visualize()
#
#     # audio = recorder.record_live(5)
#     # sample = audiosample(audio, "voice")
#     # sample.visualize()
#
#     # audio = recorder.from_file("app/resources/audio/stereo/a6.wav")
#     # sample = audiosample(audio, "file")
#     # sample.visualize()
