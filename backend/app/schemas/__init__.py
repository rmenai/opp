"""Required init file."""

from app.schemas.audio import AudioChunk, AudioChunkResponse, AudioChunkResult
from app.schemas.auth import (
    RegisterRequest,
    RegisterResponse,
    UserProfileCreate,
    UserProfileResponse,
    UserProfileUpdate,
)
from app.schemas.session import SessionCreationRequest, SessionResponse
from app.schemas.status import (
    Ack,
    HealthCheckResponse,
    Ping,
    Pong,
    ServicesStatus,
    SupabaseStatus,
    SyncRequest,
    SyncResponse,
)

__all__ = []

__all__ += ["AudioChunk", "AudioChunkResponse", "AudioChunkResult"]
__all__ += ["SessionCreationRequest", "SessionResponse"]
__all__ += [
    "Ack",
    "HealthCheckResponse",
    "Ping",
    "Pong",
    "ServicesStatus",
    "SupabaseStatus",
    "SyncRequest",
    "SyncResponse",
]
__all__ += ["RegisterRequest", "RegisterResponse", "UserProfileCreate", "UserProfileResponse", "UserProfileUpdate"]
