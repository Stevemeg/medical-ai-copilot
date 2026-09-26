"""Compatibility accessor for provider credentials from typed settings."""

from backend.settings import get_settings


class SecretNotFoundError(RuntimeError):
    pass


def get_secret(key: str, required: bool = True) -> str | None:
    if key != "GROQ_API_KEY":
        raise ValueError("Unsupported secret name")
    secret = get_settings().groq_api_key
    value = secret.get_secret_value() if secret else None
    if required and not value:
        raise SecretNotFoundError("GROQ_API_KEY is required for provider generation")
    return value


def get_groq_api_key() -> str:
    value = get_secret("GROQ_API_KEY")
    if value is None:
        raise SecretNotFoundError("GROQ_API_KEY is required for provider generation")
    return value
