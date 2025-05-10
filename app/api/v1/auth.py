import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app import schemas
from app.api.deps import get_supabase
from supabase import Client

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", response_model=schemas.RegisterResponse)
def login(req: OAuth2PasswordRequestForm = Depends(), supabase: Client = Depends(get_supabase)):
    """
    If a user with this email/password exists, log them in.
    Returns a JWT for authenticated calls.
    """
    try:
        log.debug(f"Signing in {req.username}...")
        signin = supabase.auth.sign_in_with_password({"email": req.username, "password": req.password})
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


@router.post("/register", response_model=schemas.RegisterResponse)
def register(req: OAuth2PasswordRequestForm = Depends(), supabase: Client = Depends(get_supabase)):
    """
    Create the user and then log them in.
    Must provide an email and not a username.
    Returns a JWT for authenticated calls.
    """
    try:
        log.debug(f"Registering {req.username}...")
        signup = supabase.auth.sign_up({"email": req.username, "password": req.password})
        session = signup.session
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if session is None:
        raise HTTPException(status_code=400, detail="Authentication failed")

    return schemas.RegisterResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=session.expires_in,
    )
