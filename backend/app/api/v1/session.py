"""Handle session creation."""

import datetime
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from gotrue import User

from app import schemas
from app.api.deps import get_supabase, get_user
from supabase import Client

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sessions")
def create_session(
    user: Annotated[User, Depends(get_user)],
    supabase: Annotated[Client, Depends(get_supabase)],
    session_create_request: schemas.SessionCreationRequest,
) -> schemas.SessionResponse:
    """
    Create a new audio processing session (for file or stream).

    A session is created in a 'pending' state.
    - For **file** type, provide metadata like `filename` and `content_type`.
    - For **stream** type, metadata can be optional or include expected `sample_rate`.
    """
    session_id = uuid.uuid4()
    now = datetime.datetime.now(datetime.UTC)

    session_data = {
        "session_id": str(session_id),
        "user_id": str(user.id),
        "type": session_create_request.type.value,
        "status": schemas.SessionStatus.PENDING.value,
        "metadata": session_create_request.metadata,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    log.debug("Creating session for user %s with data: %s", user.id, session_data)
    response = supabase.table("sessions").insert(session_data).execute()

    if response.data:
        created_session = response.data[0]
        return schemas.SessionResponse(**created_session)

    log.error("Failed to create session for user %s", user.id)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Could not create processing session in the database.",
    )


@router.get("/sessions")
async def list_sessions(
    user: Annotated[User, Depends(get_user)],
    supabase: Annotated[Client, Depends(get_supabase)],
    skip: Annotated[int, Query(ge=0, description="Number of records to skip for pagination")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 10,
) -> list[schemas.SessionResponse]:
    """List all processing sessions for the authenticated user (paginated)."""
    response = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", str(user.id))
        .neq("status", "closed")
        .order("created_at", desc=True)
        .offset(skip)
        .limit(limit)
        .execute()
    )

    if response.data:
        return [schemas.SessionResponse(**session_data) for session_data in response.data]

    return []  # Return empty list if no sessions or if response.data is None/empty


@router.get("/sessions/{session_id}")
async def get_session_details(
    supabase: Annotated[Client, Depends(get_supabase)],
    session_id: Annotated[uuid.UUID, Path(description="The ID of the session to retrieve")],
) -> schemas.SessionResponse:
    """Get details of a specific processing session."""
    response = supabase.table("sessions").select("*").eq("session_id", str(session_id)).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Session not found")

    return schemas.SessionResponse(**response.data[0])


@router.delete("/sessions/{session_id}")
async def delete_session(
    supabase: Annotated[Client, Depends(get_supabase)],
    session_id: Annotated[uuid.UUID, Path(description="The ID of the session to retrieve")],
) -> dict:
    """Close a specific processing session."""
    response = supabase.table("sessions").delete().eq("session_id", str(session_id)).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Failed to delete session or already deleted")

    return {"detail": "Session deleted successfully"}


@router.post("/sessions/{session_id}/close")
async def close_session(
    supabase: Annotated[Client, Depends(get_supabase)],
    session_id: Annotated[uuid.UUID, Path(description="The ID of the session to close")],
) -> dict:
    """Mark a session as closed instead of deleting it."""
    response = supabase.table("sessions").update({"status": "closed"}).eq("session_id", str(session_id)).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Session not found or already closed")

    return {"detail": "Session closed successfully"}
