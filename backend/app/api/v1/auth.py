import datetime
import logging
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from gotrue import User

from app import schemas
from app.api.deps import get_supabase, get_user
from supabase import Client, PostgrestAPIResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/auth/login", response_model=schemas.RegisterResponse)
def login(
    req: OAuth2PasswordRequestForm = Depends(), supabase: Client = Depends(get_supabase)
):
    """
    If a user with this email/password exists, log them in.
    Returns a JWT for authenticated calls.
    """
    try:
        log.debug(f"Signing in {req.username}...")
        signin = supabase.auth.sign_in_with_password(
            {"email": req.username, "password": req.password}
        )
        session = signin.session
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if session is None:
        raise HTTPException(status_code=400, detail="Authentication failed")

    return schemas.RegisterResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=session.expires_in,
    )


@router.post("/auth/register", response_model=schemas.RegisterResponse)
def register(
    req: OAuth2PasswordRequestForm = Depends(), supabase: Client = Depends(get_supabase)
):
    """
    Create the user and then log them in.
    Must provide an email and not a username.
    Returns a JWT for authenticated calls.
    """
    try:
        log.debug(f"Registering {req.username}...")
        signup = supabase.auth.sign_up(
            {"email": req.username, "password": req.password}
        )
        session = signup.session
        user = signup.user

        if user is None or session is None:
            log.warning(
                f"User object or session is None after sign_up for {req.username}. User might need email confirmation."
            )

            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Registration failed: Could not create user.",
                )

        now = datetime.datetime.now(datetime.timezone.utc)

        profile_to_create = schemas.UserProfileCreate(
            user_id=UUID(user.id), created_at=now, updated_at=now
        )
        payload = profile_to_create.model_dump(mode="json")

        log.debug(f"Creating profile for user {user.id} with data: {payload}")

        try:
            response = supabase.table("profiles").insert(payload).execute()

            if response.data is None and response.error:
                log.error(
                    f"Failed to create profile for user {user.id}: {response.error.message if response.error else 'Unknown error'}"
                )

        except Exception as e:
            log.error(
                f"Database exception while creating profile for user {user.id}: {e}"
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if session is None:
        raise HTTPException(status_code=400, detail="Registration failed")

    return schemas.RegisterResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=session.expires_in,
    )


@router.get("/me", response_model=schemas.UserProfileResponse)
async def read_user_profile(
    user: User = Depends(get_user), supabase: Client = Depends(get_supabase)
):
    """Reads the user data."""
    try:
        profile_response: PostgrestAPIResponse = (
            supabase.table("profiles")
            .select("*")
            .eq("user_id", user.id)
            .single()
            .execute()
        )

        if not profile_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found."
            )

        return schemas.UserProfileResponse(
            **profile_response.data, email=str(user.email)
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(
            f"Database exception while creating profile for user UUID {user.id}: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch user profile.",
        )

    # return schemas.UserProfileResponse(email="stop@mail.com")


@router.put("/me", response_model=schemas.UserProfileResponse)
async def update_user_profile(
    update_payload: schemas.UserProfileUpdate = Body(...),
    user: User = Depends(get_user),
    supabase: Client = Depends(get_supabase),
):
    """Updates the user data with the given information."""

    payload = update_payload.model_dump(mode="json", exclude_unset=True)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided."
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    payload["updated_at"] = now.isoformat()

    try:
        payload.pop("updated_at")
        reponse = (
            supabase.table("profiles").update(payload).eq("user_id", user.id).execute()
        )

        if not reponse:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found or update failed.",
            )

        log.info(f"Updated profile of user {user.id}")

        profile_response: PostgrestAPIResponse = (
            supabase.table("profiles")
            .select("*")
            .eq("user_id", user.id)
            .single()
            .execute()
        )

        if not profile_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found."
            )

        return schemas.UserProfileResponse(
            **profile_response.data, email=str(user.email)
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Database exception while creating profile for user {user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update user profile.",
        )
