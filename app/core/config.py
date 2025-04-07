from pydantic_settings import BaseSettings, SettingsConfigDict


class Global(BaseSettings):
    """The app settings."""

    debug: bool = False
    model_config = SettingsConfigDict(env_file=".env")


settings = Global()
