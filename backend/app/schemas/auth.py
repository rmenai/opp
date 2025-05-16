import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"


class UserProfileResponse(BaseModel):
    email: str
    language: Optional[str] = "en"
    keyboard_layout: Optional[str] = "QWERTY"


class UserProfileCreate(BaseModel):
    user_id: UUID
    created_at: datetime.datetime
    updated_at: datetime.datetime
    language: Optional[str] = "en"
    keyboard_layout: Optional[str] = "QWERTY"


class UserProfileUpdate(BaseModel):
    language: Optional[str] = "en"
    keyboard_layout: Optional[str] = "QWERTY"
