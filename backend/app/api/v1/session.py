"""Handle session creation."""

import datetime
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from gotrue import User

from app import schemas
from app.api.deps import get_supabase, get_user
from supabase import Client, PostgrestAPIResponse

log = logging.getLogger(__name__)

router = APIRouter()


async def get_session_or_404(
    session_id: uuid.UUID,
    user_id: str,
    supabase: Client,
) -> dict[str, Any]:
    """
    Fetch a session by ID for a given user_id.

    Raises HTTPException 404 if not found or not owned by the user.
    """
    session_response: PostgrestAPIResponse = (
        supabase.table("sessions")
        .select("*")
        .eq("session_id", str(session_id))
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not session_response.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or access denied.")

    return session_response.data


async def update_session_status(
    session_id: uuid.UUID,
    new_status: schemas.SessionStatus,
    supabase: Client,
) -> dict[str, Any]:
    """Update the status of a session."""
    now = datetime.datetime.now(datetime.UTC)
    (
        supabase.table("sessions")
        .update({"status": new_status.value, "updated_at": now.isoformat()})
        .eq("session_id", str(session_id))
        .execute()
    )


@router.post("/sessions")
def create_session(
    user: Annotated[User, Depends(get_user)],
    supabase: Annotated[Client, Depends(get_supabase)],
    session_create_request: schemas.SessionCreationRequest,
) -> schemas.SessionCreationResponse:
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
    response: PostgrestAPIResponse = supabase.table("sessions").insert(session_data).execute()

    if response.data:
        created_session = response.data[0]
        return schemas.SessionCreationResponse(**created_session)

    log.error("Failed to create session for user %s", user.id)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Could not create processing session in the database.",
    )


@router.get("/")
async def list_sessions(
    current_user: Annotated[User, Depends(get_user)],
    supabase: Annotated[Client, Depends(get_supabase)],
    skip: Annotated[int, Query(ge=0, description="Number of records to skip for pagination")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 10,
) -> list[schemas.SessionCreationResponse]:
    """List all processing sessions for the authenticated user (paginated)."""
    response: PostgrestAPIResponse = (
        supabase.table("sessions")
        .select("*")
        .eq("user_id", str(current_user.id))
        .order("created_at", desc=True)
        .offset(skip)
        .limit(limit)
        .execute()
    )

    if response.data:
        return [schemas.SessionCreationResponse(**session_data) for session_data in response.data]

    return []  # Return empty list if no sessions or if response.data is None/empty


@router.get("/{session_id}")
async def get_processing_session_details(
    current_user: Annotated[User, Depends(get_user)],
    supabase: Annotated[Client, Depends(get_supabase)],
    session_id: Annotated[uuid.UUID, Path(description="The ID of the session to retrieve")],
) -> schemas.SessionCreationResponse:
    """
    Get details of a specific processing session.

    Ensures the session belongs to the authenticated user.
    """
    session_data = await get_session_or_404(session_id, str(current_user.id), supabase)
    return schemas.SessionCreationResponse(**session_data)
