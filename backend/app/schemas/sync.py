from typing import Literal

from pydantic import BaseModel


class SyncRequest(BaseModel):
    client_ns: int


class SyncResponse(BaseModel):
    client_ns: int
    server_recv_ns: int
    server_send_ns: int


class Ping(BaseModel):
    type: Literal["ping"]
    client_ns: int


class Pong(BaseModel):
    type: Literal["pong"]
    client_ns: int
    server_recv_ns: int
    server_send_ns: int


class Ack(BaseModel):
    type: Literal["ack"]
    received_ns: int
