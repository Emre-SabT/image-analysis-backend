"""app/ai/embedding/ testleri - torch VE AWS GEREKTIRMEZ.

Gercek modeller (sentence-transformers / boto3) yerine sahte nesnelerle
prefiks mantigi, factory secimi/singleton'i ve boyut guard'lari dogrulanir.
"""

import numpy as np
import pytest

import app.ai.embedding as embedding_pkg
from app.ai.embedding import get_embedding_provider, reset_embedding_provider
from app.ai.embedding.bedrock_titan import BedrockTitanProvider
from app.ai.embedding.local_e5 import LocalE5Provider


class _FakeSTModel:
    """SentenceTransformer'in KULLANILAN alt kumesini taklit eder."""

    def __init__(self, dim=768):
        self._dim = dim
        self.seen: list[list[str]] = []

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **kwargs):
        self.seen.append(list(texts))
        return np.ones((len(texts), self._dim), dtype=np.float32)


def _local_provider_without_torch(fake_model):
    p = object.__new__(LocalE5Provider)  # __init__'i (torch import) ATLA
    p._model = fake_model
    return p


def test_local_e5_documents_get_passage_prefix():
    fake = _FakeSTModel()
    p = _local_provider_without_torch(fake)

    out = p.embed_documents(["deniz kenarinda aksam", "kirmizi araba"])

    assert fake.seen[-1] == ["passage: deniz kenarinda aksam", "passage: kirmizi araba"]
    assert len(out) == 2 and len(out[0]) == 768


def test_local_e5_query_gets_query_prefix():
    fake = _FakeSTModel()
    p = _local_provider_without_torch(fake)

    vec = p.embed_query("sahilde yuruyus")

    assert fake.seen[-1] == ["query: sahilde yuruyus"]
    assert len(vec) == 768


def test_local_e5_empty_documents_short_circuits():
    fake = _FakeSTModel()
    p = _local_provider_without_torch(fake)

    assert p.embed_documents([]) == []
    assert fake.seen == []  # model'e hic gidilmedi


def test_titan_rejects_unsupported_dim():
    with pytest.raises(ValueError):
        BedrockTitanProvider._validate_dim(768)  # e5 varsayilani, Titan desteklemiyor


@pytest.mark.parametrize("dim", [256, 512, 1024])
def test_titan_accepts_supported_dims(dim):
    BedrockTitanProvider._validate_dim(dim)  # firlatmamali


class _FakeProvider:
    name = "fake:test"

    def __init__(self, dim=768):
        self.dim = dim

    def embed_documents(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dim


def test_factory_is_singleton(monkeypatch):
    monkeypatch.setattr(embedding_pkg, "_build_provider", lambda: _FakeProvider())
    reset_embedding_provider()
    try:
        first = get_embedding_provider()
        second = get_embedding_provider()
        assert first is second
    finally:
        reset_embedding_provider()


def test_factory_dim_guard_rejects_mismatch(monkeypatch):
    from app.core.settings import settings

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(settings, "EMBEDDING_DIM", 999)
    monkeypatch.setattr(
        "app.ai.embedding.local_e5.LocalE5Provider", lambda: _FakeProvider(dim=768)
    )
    with pytest.raises(RuntimeError, match="EMBEDDING_DIM"):
        embedding_pkg._build_provider()


def test_factory_selects_bedrock_when_configured(monkeypatch):
    from app.core.settings import settings

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setattr(settings, "EMBEDDING_DIM", 1024)
    monkeypatch.setattr(
        "app.ai.embedding.bedrock_titan.BedrockTitanProvider",
        lambda: _FakeProvider(dim=1024),
    )
    provider = embedding_pkg._build_provider()
    assert provider.dim == 1024
