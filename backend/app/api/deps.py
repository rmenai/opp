"""FastAPI dependencies."""

import logging

from celery import Celery
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from gotrue import User
from httpx import HTTPError

from app.core import settings
from app.supabase.session import supabase
from app.worker import app as celery_app
from supabase import AuthApiError, Client

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api.endpoint}/auth/login",
    scheme_name="JWT",
)

log = logging.getLogger(__name__)


def get_supabase() -> Client:
    """Return the supabase instance, handling potential errors."""
    try:
        return supabase
    except AuthApiError as e:
        log.exception("Supabase Auth API Error: %s (Status: %s)", e.message, e.status)
        raise HTTPException(status_code=e.status or 400, detail=e.message) from e
    except HTTPError as e:
        log.exception("HTTP Error communicating with Supabase")
        raise HTTPException(status_code=502, detail="Authentication service unavailable; please try again later") from e


def get_user(token: str = Depends(oauth2_scheme)) -> User:
    """Return the authenticated user."""
    user = supabase.auth.get_user(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user.user


def get_celery() -> Celery:
    """
    Return the Celery application instance.

    The Celery app object is configured at import time.
    Errors related to broker unavailability will typically occur when trying to
    send a task, not when merely accessing this app instance.
    """
    if celery_app is None:
        log.error("Celery application instance. Check Celery initialization.")
        raise HTTPException(status_code=500, detail="Celery application not initialized.")

    return celery_app
