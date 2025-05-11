from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from gotrue import User

from app.core import settings
from app.supabase.session import supabase
from supabase import Client

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api.endpoint}/login", scheme_name="JWT")


def get_supabase() -> Client:
    """Returns the supabase instance."""
    return supabase


def get_user(token: str = Depends(oauth2_scheme)) -> User:
    """Return the authenticated user."""
    user = supabase.auth.get_user(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user.user
