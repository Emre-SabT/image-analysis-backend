"""Metin embedding SOYUTLAMASI (semantik arama).

Amac: semantic_service.py'nin hangi embedding motorunu (yerel e5 / AWS
Bedrock Titan) kullandigini bilmeden calismasini saglamak - saglayici
karari TEK noktaya (get_embedding_provider factory'si, __init__.py) ertelenir.
candidate_search.CandidateFinder ile AYNI desen (Protocol + factory).

ASIMETRI NOTU: e5 ailesi girdi ONEKI ister (dokumanlar "passage: ", sorgular
"query: "); Titan onek KULLANMAZ. Bu ayrimi HER Provider kendi embed_documents
/ embed_query metodunda saklar - cagiran (semantic_service) ham metin verir.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """VLM JSON analizinden uretilen dokumani / arama sorgusunu vektore cevirir.

    name: "provider:model" kimligi - photo_embeddings.model kolonuna yazilir
          (backfill'in "farkli modelle indekslenmis olanlari yeniden isle"
          sorgusu buna bakar).
    dim:  uretilen vektorun boyutu - Qdrant 'photo_semantic' koleksiyonu bu
          boyutta olusturulur; get_embedding_provider() bunun
          settings.EMBEDDING_DIM ile ESIT oldugunu dogrular.
    """

    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Birden cok dokumani (pasaj) vektorler. Bos liste -> bos liste."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Tek bir arama sorgusunu vektorler."""
        ...
