from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Global(BaseSettings):
    """The app settings."""

    debug: bool = False
    resources_dir: Path = Path("resources/")
    model_config = SettingsConfigDict(env_file=".env")


settings = Global()
