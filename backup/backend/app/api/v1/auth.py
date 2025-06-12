"""Handle user authentication."""

import datetime
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from gotrue import User

from app import schemas
from app.api.deps import get_supabase, get_user
from supabase import Client

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/auth/login")
def login(
    req: Annotated[OAuth2PasswordRequestForm, Depends()],
    supabase: Annotated[Client, Depends(get_supabase)],
) -> schemas.RegisterResponse:
    """
    If a user with this email/password exists, log them in.

    Returns a JWT for authenticated calls.
    """
    log.debug("Signing in %s...", req.username)
    signin = supabase.auth.sign_in_with_password({"email": req.username, "password": req.password})
    session = signin.session

    if session is None:
        raise HTTPException(status_code=400, detail="Authentication failed")

    return schemas.RegisterResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=session.expires_in,
    )


@router.post("/auth/register")
def register(
    req: Annotated[OAuth2PasswordRequestForm, Depends()],
    supabase: Annotated[Client, Depends(get_supabase)],
) -> schemas.RegisterResponse:
    """
    Create the user and then log them in.

    Must provide an email and not a username.
    Returns a JWT for authenticated calls.
    """
    log.debug("Registering %s...", req.username)
    signup = supabase.auth.sign_up({"email": req.username, "password": req.password})
    session = signup.session
    user = signup.user

    if user is None or session is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Registration failed.")

    now = datetime.datetime.now(datetime.UTC)

    profile_to_create = schemas.UserProfileCreate(
        user_id=UUID(user.id),
        created_at=now,
        updated_at=now,
    )

    payload = profile_to_create.model_dump(mode="json")

    log.debug("Creating profile for user %s with data: %s", user.id, payload)

    response = supabase.table("profiles").insert(payload).execute()
    if not response:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile creation failed.")

    if session is None:
        raise HTTPException(status_code=400, detail="Registration failed")

    return schemas.RegisterResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=session.expires_in,
    )


@router.get("/me")
async def read_user_profile(
    user: Annotated[User, Depends(get_user)],
    supabase: Annotated[Client, Depends(get_supabase)],
) -> schemas.UserProfileResponse:
    """Read the user data."""
    response = supabase.table("profiles").select("*").eq("user_id", user.id).single().execute()
    if not response.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

    return schemas.UserProfileResponse(
        **response.data,
        email=str(user.email),
    )


@router.put("/me")
async def update_user_profile(
    user: Annotated[User, Depends(get_user)],
    supabase: Annotated[Client, Depends(get_supabase)],
    update_payload: Annotated[schemas.UserProfileUpdate, Body()] = ...,
) -> schemas.UserProfileResponse:
    """Update the user data with the given information."""
    payload = update_payload.model_dump(mode="json", exclude_unset=True)
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided.")

    now = datetime.datetime.now(datetime.UTC)
    payload["updated_at"] = now.isoformat()

    response = supabase.table("profiles").update(payload).eq("user_id", user.id).execute()
    if not response:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found or update failed.")

    log.info("Updated profile of user %s", user.id)

    profile_response = supabase.table("profiles").select("*").eq("user_id", user.id).single().execute()
    if not profile_response.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

    return schemas.UserProfileResponse(
        **profile_response.data,
        email=str(user.email),
    )
