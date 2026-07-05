from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ENV_FILE = Path(__file__).resolve().parents[5] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPOSITORY_ENV_FILE, ".env"),
        env_prefix="APP_",
        extra="ignore",
    )

    environment: str = "development"
    model_id: str = "google/gemma-4-E4B-it"
    allowed_origins_csv: str = "http://localhost:3000"
    max_upload_mb: int = 15
    sarvam_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="SARVAM_API_KEY",
    )

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins_csv.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
