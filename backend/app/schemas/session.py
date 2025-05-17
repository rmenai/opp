"""Schemas for recording sessions."""

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SessionType(str, Enum):
    """Available session types."""

    FILE = "file"
    STREAM = "stream"


class SessionStatus(str, Enum):
    """Available session statuses."""

    PENDING = "pending"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Stream ended, some results might be available


class SessionCreationRequest(BaseModel):
    """Session creation request."""

    type: SessionType
    metadata: dict[str, Any] | None = Field(
        None,
        description="e.g., {'filename': 'my_audio.wav', 'sample_rate': 44100} for file type",
    )


class SessionCreationResponse(BaseModel):
    """Session creation response."""

    session_id: str
    user_id: str
    type: SessionType
    status: SessionStatus = SessionStatus.PENDING
    metadata: dict[str, Any] | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
