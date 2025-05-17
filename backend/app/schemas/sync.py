"""Schemas for syncing time."""

from typing import Literal

from pydantic import BaseModel


class SyncRequest(BaseModel):
    """SyncRequest."""

    client_ns: int


class SyncResponse(BaseModel):
    """SyncResponse."""

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
