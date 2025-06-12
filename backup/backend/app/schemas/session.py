"""Schemas for recording sessions."""

import datetime
import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.models.session import SessionStatus, SessionType


class SessionCreationRequest(BaseModel):
    """Session creation request."""

    type: SessionType
    metadata: dict[str, Any] | None = Field(
        None,
        description="e.g., {'filename': 'my_audio.wav', 'sample_rate': 44100} for file type",
    )


class SessionResponse(BaseModel):
    """Session creation response."""

    session_id: uuid.UUID
    user_id: uuid.UUID
    type: SessionType
    status: SessionStatus = SessionStatus.PENDING
    metadata: dict[str, Any] | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
