"""Required init file."""

from app.worker.celery import app

__all__ = ["app"]
