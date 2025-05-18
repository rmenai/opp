"""Settings for tests."""

from app.core import settings

BASE_URL = f"http://{settings.api.host}:{settings.api.port}"
BASE_API_URL = f"{BASE_URL}/{settings.api.endpoint.strip('/')}"
