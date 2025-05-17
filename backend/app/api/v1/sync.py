"""Sync the clock with the client's."""

import time

from fastapi import APIRouter, WebSocket

from app import schemas

router = APIRouter()


@router.post("/sync")
def sync(req: schemas.SyncRequest) -> schemas.SyncResponse:
    """
    Sync the clock between the client and the server.

    T1 = client send time
    T2 = server receive time
    T3 = server send time
    T4 = client receive time

    offset = ((T2 - T1) + (T3 - T4)) / 2
    rtt    = (T4 - T1) - (T3 - T2)
    """
    server_recv = time.time_ns()
    server_send = time.time_ns()

    return schemas.SyncResponse(
        client_ns=req.client_ns,
        server_recv_ns=server_recv,
        server_send_ns=server_send,
    )


@router.websocket("/ws/sync")
async def ws_sync(ws: WebSocket) -> None:
    """Sync the clock between the client and the server continuously."""
    await ws.accept()

    while True:
        data = await ws.receive_json()
        t2 = time.time_ns()
        ping = schemas.Ping.model_validate(data)

        t3 = time.time_ns()
        pong = schemas.Pong(
            type="pong",
            client_ns=ping.client_ns,
            server_recv_ns=t2,
            server_send_ns=t3,
        )

        await ws.send_json(pong.model_dump())
