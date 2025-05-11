from fastapi import APIRouter

from app.api.v1 import audio, auth, sync

api_router = APIRouter()

api_router.include_router(audio.router)
api_router.include_router(auth.router)
api_router.include_router(sync.router)
