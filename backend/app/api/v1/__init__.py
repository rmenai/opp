from fastapi import APIRouter

from app.api.v1 import auth, session, sync

api_router = APIRouter()

api_router.include_router(session.router)
api_router.include_router(auth.router)
api_router.include_router(sync.router)
