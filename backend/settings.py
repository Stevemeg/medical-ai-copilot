"""Validated process configuration. Production has no implicit development mode."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["development", "test", "production"]
    database_url: str
    auth_mode: Literal["dev", "oidc"]
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    dev_jwt_key: SecretStr | None = None
    audit_hmac_key: SecretStr
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    groq_max_retries: int = Field(default=1, ge=0, le=3)
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cors_origins: list[str] = ["http://localhost:5500", "http://localhost:18000", "http://localhost:5173"]
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    db_pool_timeout: int = Field(default=10, ge=1, le=120)
    rate_limit_per_minute: int = Field(default=20, ge=1)
    max_fhir_body_bytes: int = Field(default=2_000_000, ge=1024)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def secure_modes(self) -> "Settings":
        if not self.database_url.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must use PostgreSQL and psycopg")
        if self.app_env == "production":
            if self.auth_mode != "oidc" or not all((self.oidc_issuer, self.oidc_audience, self.oidc_jwks_url)):
                raise ValueError("Production requires OIDC issuer, audience and JWKS URL")
            if not (self.oidc_issuer or "").startswith("https://") or not (self.oidc_jwks_url or "").startswith(
                "https://"
            ):
                raise ValueError("Production OIDC issuer and JWKS URL must use HTTPS")
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS is forbidden in production")
            if self.audit_hmac_key.get_secret_value().lower().startswith("replace-"):
                raise ValueError("Production requires a real audit key")
        if len(self.audit_hmac_key.get_secret_value()) < 32:
            raise ValueError("Audit key must contain at least 32 characters")
        if self.auth_mode == "oidc" and not all((self.oidc_issuer, self.oidc_audience, self.oidc_jwks_url)):
            raise ValueError("OIDC requires issuer, audience and JWKS URL")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values are read from environment by BaseSettings
