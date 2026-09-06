"""Embedding saglayici factory'si - candidate_search.get_candidate_finder ile
AYNI desen, tek farki: SUREC BASINA TEK ORNEK (singleton).

Yerel e5 icin model ilk cagrida GPU'ya yuklenir (~1.1 GB); sonraki tum
cagrilar ayni ornegi kullanir (bkz. dispatcher._get_bedrock_client). Testler
reset_embedding_provider() ile singleton'i sifirlar ya da _provider'a dogrudan
bir sahte atar.
"""

from __future__ import annotations

import logging

from app.ai.embedding.base import EmbeddingProvider
from app.core.settings import settings

logger = logging.getLogger("photoai.embedding")

_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = _build_provider()
    return _provider


def reset_embedding_provider() -> None:
    """Testler icin - bir sonraki get_embedding_provider() yeniden kurar."""
    global _provider
    _provider = None


def _build_provider() -> EmbeddingProvider:
    name = settings.EMBEDDING_PROVIDER  # settings validator: 'local' | 'bedrock'
    if name == "bedrock":
        from app.ai.embedding.bedrock_titan import BedrockTitanProvider

        provider: EmbeddingProvider = BedrockTitanProvider()
    else:
        from app.ai.embedding.local_e5 import LocalE5Provider

        provider = LocalE5Provider()

    if provider.dim != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding provider boyutu ({provider.dim}) ile ayar EMBEDDING_DIM "
            f"({settings.EMBEDDING_DIM}) uyusmuyor - Qdrant 'photo_semantic' "
            f"koleksiyonu EMBEDDING_DIM'e gore olusturuluyor, bu deger provider'in "
            f"gercek ciktisiyla ESIT olmali."
        )
    logger.info("Embedding provider secildi: %s (dim=%d)", provider.name, provider.dim)
    return provider
