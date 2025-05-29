"""Degine User model."""

import datetime
import uuid

from sqlmodel import Field, SQLModel


class Profile(SQLModel, table=True):
    """Profile sql."""

    user_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    language: str | None = Field(default="en")
    keyboard_layout: str | None = Field(default="QWERTY")
