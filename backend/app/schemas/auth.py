"""Schemas for handling authentication."""

import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    """Register request."""

    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    """Register response."""

    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"  # noqa: S105


class UserProfileResponse(BaseModel):
    """User profile response."""

    email: str
    language: str | None = "en"
    keyboard_layout: str | None = "QWERTY"


class UserProfileCreate(BaseModel):
    """User profile create."""

    user_id: UUID
    created_at: datetime.datetime
    updated_at: datetime.datetime
    language: str | None = "en"
    keyboard_layout: str | None = "QWERTY"


class UserProfileUpdate(BaseModel):
    """User profile update."""

    language: str | None = "en"
    keyboard_layout: str | None = "QWERTY"
