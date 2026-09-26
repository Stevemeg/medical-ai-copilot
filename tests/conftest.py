"""Offline legacy unit tests use an explicit test configuration."""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("AUTH_MODE", "dev")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://medical:test@localhost:5432/medical")
os.environ.setdefault("AUDIT_HMAC_KEY", "unit-test-only-audit-key-do-not-use-in-deployments")
os.environ.setdefault("DEV_JWT_KEY", "unit-test-only-jwt-key-do-not-use-in-deployments-0123456789012345")
