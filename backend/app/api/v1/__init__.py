"""Route all routes to the app."""

from fastapi import APIRouter

from app.api.v1 import auth, session, status

api_router = APIRouter()

api_router.include_router(session.router)
api_router.include_router(auth.router)
api_router.include_router(status.router)
