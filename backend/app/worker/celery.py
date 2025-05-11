import logging

from celery import Celery

from app.core import settings

log = logging.getLogger(__name__)

app = Celery(
    "audio",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

log.info("Initialized Celery worker")
