"""GET /photos/search rota handler'i (search_photos).

Kod tabaninda HTTP-seviyesi test altyapisi (TestClient) yok - diger router
testleri gibi handler DOGRUDAN cagrilir. `db` parametresi govdede
kullanilmiyor (yalnizca Depends icin), None gecilir.

Asil arama mantigi (semantic_service.search + Qdrant) test_semantic_service.py'de
uctan uca test ediliyor; burada YALNIZCA rota sozlesmesi dogrulanir:
flag -> 503, hata -> 502, sekil, limit kirpma.
"""

import pytest
from fastapi import HTTPException

from app.core.settings import settings
from app.routers.photos import search_photos
from app.services import semantic_service


def test_returns_503_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_SEARCH_ENABLED", False)
    with pytest.raises(HTTPException) as exc:
        search_photos(q="deniz kenari", limit=50, db=None)
    assert exc.value.status_code == 503


def test_returns_502_when_search_raises(monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_SEARCH_ENABLED", True)

    def boom(q, limit, min_score):
        raise RuntimeError("qdrant erisilemez")

    monkeypatch.setattr(semantic_service, "search", boom)
    with pytest.raises(HTTPException) as exc:
        search_photos(q="deniz kenari", limit=50, db=None)
    assert exc.value.status_code == 502
    assert "RuntimeError" in exc.value.detail


def test_happy_path_returns_photo_id_score_shape(monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_SEARCH_ENABLED", True)
    monkeypatch.setattr(
        semantic_service,
        "search",
        lambda q, limit, min_score: [("p1", 0.91), ("p2", 0.42)],
    )
    out = search_photos(q="deniz kenari", limit=50, db=None)
    assert out == [
        {"photo_id": "p1", "score": 0.91},
        {"photo_id": "p2", "score": 0.42},
    ]


def test_min_score_from_settings_is_forwarded(monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "SEMANTIC_SEARCH_MIN_SCORE", 0.5)
    seen = {}

    def capture(q, limit, min_score):
        seen["min_score"] = min_score
        return []

    monkeypatch.setattr(semantic_service, "search", capture)
    search_photos(q="x", limit=50, db=None)
    assert seen["min_score"] == 0.5


def test_limit_is_clamped(monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_SEARCH_ENABLED", True)
    monkeypatch.setattr(settings, "SEMANTIC_SEARCH_TOP_K", 200)
    seen = {}

    def capture(q, limit, min_score):
        seen["limit"] = limit
        return []

    monkeypatch.setattr(semantic_service, "search", capture)

    search_photos(q="x", limit=99999, db=None)
    assert seen["limit"] == 200  # ust sinir

    search_photos(q="x", limit=0, db=None)
    assert seen["limit"] == 1  # alt sinir
