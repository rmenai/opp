import glob
from pathlib import Path

from pydub import AudioSegment


def get_audio_files(pattern="audio/*.wav"):
    return [Path(file) for file in glob.glob(pattern)]


def convert_to_mono(file: Path, output_dir="audio/mono"):
    print(f"Converting {file} to mono")
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    output_file = Path(output_dir) / file.name
    sound.export(str(output_file), format="wav")


def convert_to_stereo(file: Path, output_dir="audio/stereo"):
    print(f"Converting {file} to stereo")
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(2)
    output_file = Path(output_dir) / file.name
    sound.export(str(output_file), format="wav")


def process_all_files():
    all = get_audio_files("audio/*.wav")
    mono = {file.name for file in get_audio_files("audio/mono/*.wav")}
    stereo = {file.name for file in get_audio_files("audio/stereo/*.wav")}

    for file in all:
        if file.name not in mono:
            convert_to_mono(file)

        if file.name not in stereo:
            convert_to_stereo(file)
