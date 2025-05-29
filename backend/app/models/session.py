"""Degine User model."""

import datetime
import uuid
from enum import Enum

from sqlmodel import Field, SQLModel


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
    CLOSED = "closed"


class Session(SQLModel, table=True):
    """User sql."""

    session_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    user_id: uuid.UUID = Field(index=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    type: SessionType = Field()
    status: SessionStatus = Field(default=SessionStatus.PENDING)
