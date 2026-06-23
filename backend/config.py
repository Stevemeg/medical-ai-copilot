"""
Secrets and configuration loading.

DESIGN: this checks environment variables FIRST, falling back to
Streamlit's secrets.toml only for local development convenience. This
matters because environment variables are the actual mechanism every real
cloud secrets manager uses to deliver secrets to a running application --
AWS Secrets Manager, Azure Key Vault, and GCP Secret Manager don't hand
your code a secret directly; they're typically read once at deploy/
startup time and injected into the process environment (or fetched via
their SDK and then set as an env var), and the application code just
reads os.environ like it would anywhere else.

This means the SAME code here works unchanged whether you're running
locally with a secrets.toml file, or deployed somewhere that injects
GROQ_API_KEY as a real environment variable from a real secrets manager --
nothing in backend/rag_pipeline.py needs to know or care which one is
happening.

Production migration path (for when real cloud infrastructure is
available):

  AWS:   aws secretsmanager get-secret-value --secret-id <name>
         then export the result as GROQ_API_KEY before starting the app
         (or use AWS's --secrets argument in ECS task definitions, which
         injects it as an env var automatically).

  Azure: az keyvault secret show --vault-name <vault> --name GROQ-API-KEY
         Azure App Service has a direct "Key Vault reference" app setting
         that injects secrets as env vars without any code changes.

  GCP:   gcloud secrets versions access latest --secret=GROQ_API_KEY
         Cloud Run has a direct "Reference a secret" option in its
         environment variables configuration that does the same thing.

In all three cases, the actual code reading the secret (this file) does
not change -- only how the environment variable gets populated changes,
which is exactly the point of separating secret STORAGE from secret USE.
"""

import os


class SecretNotFoundError(Exception):
    """Raised when a required secret isn't available from any source."""
    pass


def get_secret(key: str, required: bool = True):
    """
    Resolves a secret by name, checking sources in this order:
      1. Environment variable (the real production path)
      2. Streamlit's st.secrets (local development convenience)

    Returns None if not found and required=False, otherwise raises
    SecretNotFoundError with a message indicating both sources were
    checked, so a misconfiguration is obvious rather than silent.
    """
    value = os.environ.get(key)
    if value:
        return value

    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # st.secrets raises if secrets.toml doesn't exist at all, or if
        # called outside a Streamlit runtime context (e.g. a future
        # non-Streamlit deployment) -- either way, fall through to the
        # "not found" handling below rather than crash here.
        pass

    if required:
        raise SecretNotFoundError(
            f"'{key}' not found in environment variables or "
            f".streamlit/secrets.toml. Set it as an environment variable "
            f"(the production path) or add it to secrets.toml (local dev)."
        )
    return None


def get_groq_api_key() -> str:
    return get_secret("GROQ_API_KEY")