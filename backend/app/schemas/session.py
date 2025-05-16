import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SessionType(str, Enum):
    FILE = "file"
    STREAM = "stream"


class SessionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Stream ended, some results might be available


class SessionCreationRequest(BaseModel):
    type: SessionType
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="e.g., {'filename': 'my_audio.wav', 'sample_rate': 44100} for file type",
    )


class SessionCreationResponse(BaseModel):
    session_id: str
    user_id: str
    type: SessionType
    status: SessionStatus = SessionStatus.PENDING
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        from_attributes = True
