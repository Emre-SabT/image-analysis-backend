"""Yerel embedding saglayici: intfloat/multilingual-e5-base (VARSAYILAN).

GPU'da calisir (settings.EMBEDDING_DEVICE, varsayilan "cuda") - VLM AWS
Bedrock'a tasindigi icin yerel GPU bosta. Model SUREC BASINA BIR KEZ
yuklenir (bkz. get_embedding_provider singleton'i, __init__.py); ~1.1 GB
agirlik, 768-d cikti.

e5 ONEK KURALI burada gizli:
  - dokumanlar  -> "passage: " + metin
  - sorgular    -> "query: "   + metin
normalize_embeddings=True: kosinus benzerligi icin birim norm (Qdrant
'photo_semantic' koleksiyonu da Cosine mesafeyle olusturuluyor).
"""

from __future__ import annotations

import logging

from app.core.settings import settings

logger = logging.getLogger("photoai.embedding")

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "

# e5-base sabit boyut - yanlis model dizini verilirse __init__ bunu yakalar.
_E5_BASE_DIM = 768


class LocalE5Provider:
    name = "local:intfloat/multilingual-e5-base"
    dim = _E5_BASE_DIM

    def __init__(self) -> None:
        # Agir import (torch) yalnizca bu saglayici GERCEKTEN secilince yapilir -
        # EMBEDDING_PROVIDER=bedrock kurulumlar sentence-transformers/torch
        # kurmak zorunda kalmasin.
        from sentence_transformers import SentenceTransformer

        logger.info(
            "e5 modeli yukleniyor: dir=%s device=%s",
            settings.EMBEDDING_MODEL_DIR,
            settings.EMBEDDING_DEVICE,
        )
        self._model = SentenceTransformer(
            settings.EMBEDDING_MODEL_DIR, device=settings.EMBEDDING_DEVICE
        )
        # sentence-transformers 6.0'da get_sentence_embedding_dimension ->
        # get_embedding_dimension olarak yeniden adlandirildi; ikisini de destekle.
        _get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        actual = _get_dim()
        if actual != self.dim:
            raise RuntimeError(
                f"e5 model boyutu {actual}, beklenen {self.dim} - "
                f"yanlis model dizini olabilir ({settings.EMBEDDING_MODEL_DIR})."
            )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([_PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([_QUERY_PREFIX + text])[0]
