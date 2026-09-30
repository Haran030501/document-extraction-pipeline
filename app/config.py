from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg:///extraction?host=/var/run/postgresql"
    anthropic_api_key: str | None = None
    model: str = "claude-opus-5-5"
    # Set explicitly: defaults differ by model (medium on Opus 5.5, high on Sonnet 5.5).
    effort: str = "high"
    prompt_version: str = "v3"
    max_retries: int = 2
    # Pages with fewer extracted characters than this are treated as scanned and OCR'd.
    ocr_min_chars: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
