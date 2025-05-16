import datetime
import logging
import time
import uuid
from typing import Annotated, Any, Dict, List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from gotrue import User

from app import schemas
from app.api.deps import get_supabase, get_user
from supabase import Client, PostgrestAPIResponse

log = logging.getLogger(__name__)

router = APIRouter()


async def get_session_or_404(
    session_id: uuid.UUID, user_id: str, supabase: Client
) -> Dict[str, Any]:
    """
    Fetches a session by ID for a given user_id.
    Raises HTTPException 404 if not found or not owned by the user.
    """
    try:
        session_response: PostgrestAPIResponse = (
            supabase.table("sessions")
            .select("*")
            .eq("session_id", str(session_id))
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not session_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or access denied.",
            )
        return session_response.data

    except HTTPException:
        raise

    except Exception as e:
        log.error(
            f"Database error fetching session {session_id} for user {user_id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch session.",
        )


async def update_session_status(
    session_id: uuid.UUID, new_status: schemas.SessionStatus, supabase: Client
) -> Dict[str, Any]:
    """
    Updates the status of a session.
    """
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        updated_session_response: PostgrestAPIResponse = (
            supabase.table("sessions")
            .update({"status": new_status.value, "updated_at": now.isoformat()})
            .eq("session_id", str(session_id))
            .execute()
        )
        if not updated_session_response.data:
            log.warning(
                f"Attempted to update status for session {session_id} but no rows were affected or data was empty."
            )
            current_session = await get_session_or_404(session_id, "", supabase)
            if not current_session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Session not found for status update.",
                )
            return current_session

        # Assuming the update returns the updated record(s); Supabase often returns a list.
        # If it returns a list, and we expect one record:
        if (
            isinstance(updated_session_response.data, list)
            and len(updated_session_response.data) > 0
        ):
            return updated_session_response.data[0]
        elif isinstance(
            updated_session_response.data, dict
        ):  # If it returns a single dict
            return updated_session_response.data
        else:
            log.error(
                f"Unexpected response format from Supabase update for session {session_id}: {updated_session_response.data}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update session status due to unexpected response.",
            )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Database error updating session {session_id} status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update session status.",
        )


@router.post("/sessions", response_model=schemas.SessionCreationResponse)
def create_session(
    session_create_request: schemas.SessionCreationRequest,
    user: User = Depends(get_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Create a new audio processing session (for file or stream).

    A session is created in a 'pending' state.
    - For **file** type, provide metadata like `filename` and `content_type`.
    - For **stream** type, metadata can be optional or include expected `sample_rate`.
    """
    session_id = uuid.uuid4()
    now = datetime.datetime.now(datetime.timezone.utc)

    session_data = {
        "session_id": str(session_id),
        "user_id": str(user.id),
        "type": session_create_request.type.value,
        "status": schemas.SessionStatus.PENDING.value,
        "metadata": session_create_request.metadata,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    try:
        log.debug(f"Creating session for user {user.id} with data: {session_data}")
        response: PostgrestAPIResponse = (
            supabase.table("sessions").insert(session_data).execute()
        )

        if response.data:
            created_session = response.data[0]
            return schemas.SessionCreationResponse(**created_session)
        else:
            log.error(f"Failed to create session for user {user.id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create processing session in the database.",
            )

    except Exception as e:
        log.error(f"Exception creating session for user {user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.get("/", response_model=List[schemas.SessionCreationResponse])
async def list_sessions(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of records to return"
    ),
    current_user: User = Depends(get_user),
    supabase: Client = Depends(get_supabase),
):
    """
    List all processing sessions for the authenticated user (paginated).
    """
    try:
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
            return [
                schemas.SessionCreationResponse(**session_data)
                for session_data in response.data
            ]
        return []  # Return empty list if no sessions or if response.data is None/empty
    except Exception as e:
        log.error(f"Error listing sessions for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve sessions.",
        )


@router.get("/{session_id}", response_model=schemas.SessionCreationResponse)
async def get_processing_session_details(
    session_id: uuid.UUID = Path(..., description="The ID of the session to retrieve"),
    current_user: User = Depends(get_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Get details of a specific processing session.
    Ensures the session belongs to the authenticated user.
    """
    session_data = await get_session_or_404(session_id, str(current_user.id), supabase)
    return schemas.SessionCreationResponse(**session_data)
