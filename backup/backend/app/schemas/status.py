"""Schemas for syncing time."""

from typing import Literal

from pydantic import BaseModel


class SupabaseStatus(BaseModel):
    """Supabase status."""

    db: Literal["ok", "error"]
    auth: Literal["ok", "error"]
    storage: Literal["ok", "error"]


class ServicesStatus(BaseModel):
    """Service status."""

    supabase: SupabaseStatus
    celery: Literal["ok", "error"]


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: Literal["ok", "error"]
    services: ServicesStatus


class SyncRequest(BaseModel):
    """Sync request."""

    client_ns: int


class SyncResponse(BaseModel):
    """Sync response."""

    client_ns: int
    server_recv_ns: int
    server_send_ns: int


class Ping(BaseModel):
    """Ping."""

    type: Literal["ping"]
    client_ns: int


class Pong(BaseModel):
    """Pong."""

    type: Literal["pong"]
    client_ns: int
    server_recv_ns: int
    server_send_ns: int


class Ack(BaseModel):
    """Ack."""

    type: Literal["ack"]
    received_ns: int
