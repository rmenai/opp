from typing import Literal

from pydantic import BaseModel


class AudioChunk(BaseModel):
    type: Literal["audio_chunk"]
    client_id: str
    timestamp_ns: int
    pcm_data_b64: str


class AudioData(BaseModel):
    user_id: str
