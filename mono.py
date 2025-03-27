import glob
from pathlib import Path

from pydub import AudioSegment

all = [Path(file) for file in glob.glob("audio/*.wav")]
mono = [Path(file) for file in glob.glob("audio/mono/*.wav")]
stereo = [Path(file) for file in glob.glob("audio/stereo/*.wav")]

for file in all:
    if not file.name in [f.name for f in mono]:
        print(f"Converting {file} to mono")

        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export(f"audio/mono/{file.name}", format="wav")

    if not file.name in [f.name for f in stereo]:
        print(f"Converting {file} to stereo")

        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(2)
        sound.export(f"audio/stereo/{file.name}", format="wav")
