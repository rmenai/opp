"""Sync the clock with the client's."""

import logging
import time
from typing import Annotated, Literal

from celery import Celery
from fastapi import APIRouter, Depends, Response, WebSocket, status

from app import schemas
from app.api.deps import get_celery, get_supabase
from supabase import Client

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/healthz")
async def health_check(
    supabase: Annotated[Client, Depends(get_supabase)],
    celery: Annotated[Celery, Depends(get_celery)],
    response: Response,
) -> schemas.HealthCheckResponse:
    """Health check endpoint that verifies connectivity to Supabase and Celery."""
    celery_status: Literal["ok", "error"] = "ok"

    db_status: Literal["ok", "error"] = "ok"
    auth_status: Literal["ok", "error"] = "ok"
    storage_status: Literal["ok", "error"] = "ok"

    try:
        supabase.table("profiles").select("user_id").limit(1).execute()  # Database
    except Exception:
        db_status = "error"
        log.exception("Supabase DB health check exception")

    try:
        supabase.auth.admin.list_users()  # Auth
    except Exception:
        auth_status = "error"
        log.exception("Supabase Auth health check exception")

    try:
        supabase.storage.list_buckets()  # Storage
    except Exception:
        storage_status = "error"
        log.exception("Supabase Storage health check exception")

    try:
        responses = celery.control.ping(timeout=1)
        if not responses:
            celery_status = "error"
            log.error("No Celery workers responded to ping")
    except Exception:
        celery_status = "error"
        log.exception("Celery health check exception")

    overall = "error" if "error" in (celery_status, db_status, auth_status, storage_status) else "ok"
    if overall == "error":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return schemas.HealthCheckResponse(
        status=overall,
        services=schemas.ServicesStatus(
            supabase=schemas.SupabaseStatus(
                db=db_status,
                auth=auth_status,
                storage=storage_status,
            ),
            celery=celery_status,
        ),
    )


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
