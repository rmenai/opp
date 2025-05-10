from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class API(BaseSettings):
    """The API settings."""

    name: str = "FastAPI"
    endpoint: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8080

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="allow")


class SupaBase(BaseSettings):
    """The SupaBase settings."""

    url: str = ""
    key: SecretStr = SecretStr("")

    model_config = SettingsConfigDict(env_prefix="SUPABASE_", env_file=".env", extra="allow")


class Celery(BaseSettings):
    """The Celery settings/"""

    broker_url: str = "redis://127.0.0.1:6379/0"
    result_backend: str = "redis://127.0.0.1:6379/0"


class Settings(BaseSettings):
    """The app settings."""

    debug: bool = False
    audio_dir: Path = Path("app/resources/audio/native/")

    api: API = API()
    supabase: SupaBase = SupaBase()
    celery: Celery = Celery()

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()


settings = get_settings()
