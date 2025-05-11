import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket
from gotrue import User

from app import schemas
from app.api.deps import get_supabase, get_user

router = APIRouter()


@router.get("/me")
def me(user: User = Depends(get_user)):
    return {
        "id": user.id,
        "email": user.email,
    }
