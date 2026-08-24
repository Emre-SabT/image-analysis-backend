"""Madde 3: IDENTITY_SEARCH_BACKEND=pg_brute_force, PR-D bitene kadar
UYGULAMA BASLANGICINDA (Settings() insasi aninda) reddedilmeli."""

import pytest
from pydantic import ValidationError

from app.core.settings import Settings


def test_pg_brute_force_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("IDENTITY_SEARCH_BACKEND", "pg_brute_force")
    with pytest.raises(ValidationError, match="PR-D bitene kadar"):
        Settings()


def test_qdrant_default_is_accepted():
    s = Settings()
    assert s.IDENTITY_SEARCH_BACKEND == "qdrant"


def test_invalid_backend_value_is_rejected(monkeypatch):
    monkeypatch.setenv("IDENTITY_SEARCH_BACKEND", "something_else")
    with pytest.raises(ValidationError):
        Settings()
