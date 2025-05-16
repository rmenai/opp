from typing import Optional

from pydantic import BaseModel, Field


class AudioChunk(BaseModel):
    content_type: Optional[str] = Field(
        "audio/webm;codecs=opus",
        description="MIME type of the audio chunk, e.g., 'audio/wav', 'audio/webm;codecs=opus'",
    )
    audio_chunk_b64: str = Field(..., description="Base64 encoded audio data chunk.")
    timestamp_utc: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of the chunk start. e.g., '2023-10-27T10:30:00.123Z'",
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Monotonically increasing sequence number for ordering chunks, starting from 0.",
    )
    is_last_chunk: bool = Field(
        default=False, description="Indicates if this is the last chunk in a stream."
    )


class AudioChunkResponse(BaseModel):
    status: str = Field(
        description="e.g., 'received', 'error_invalid_format', 'error_sequence_mismatch'"
    )
    received_sequence_number: int
    session_id: str


class AudioChunkResult(BaseModel):
    """Chunk containing all coefficients data"""

    pass
