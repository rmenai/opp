"""Schemas for binary audio."""

from pydantic import BaseModel, Field


class AudioChunk(BaseModel):
    """Audio chunk."""

    content_type: str | None = Field(default="audio/webm;codecs=opus", description="MIME type of chunk")
    audio_chunk_b64: str = Field(..., description="Base64-encoded audio data")
    timestamp_utc: str = Field(..., description="ISO8601 UTC timestamp of chunk start")
    sequence_number: int = Field(..., ge=0, description="Ordering sequence number (>=0)")
    is_last_chunk: bool = Field(default=False, description="True if this is the final chunk")


class AudioChunkResponse(BaseModel):
    """Audio chunk response."""

    status: str = Field(description="e.g., 'received', 'error_invalid_format', 'error_sequence_mismatch'")
    received_sequence_number: int
    session_id: str


class AudioChunkResult(BaseModel):
    """Chunk containing all coefficients data."""
