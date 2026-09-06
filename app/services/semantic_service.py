"""Semantik arama - VLM JSON analizini (photo_analysis) metin embedding'ine
cevirip Qdrant 'photo_semantic' koleksiyonunda indeksler ve sorgular.

Akis (bkz. konusma gecmisindeki diyagram):

    photo_analysis  --build_document-->  tek metin dokumani
                    --EmbeddingProvider-->  1 vektor
                    --qdrant.upsert-->      photo_semantic  (point_id = photo_id)
                    + photo_embeddings satiri (model/dim/indexed_at - izleme)

    q  --embed_query-->  vektor  --qdrant.query_points-->  sirali [(photo_id, score)]

KAYNAK-OF-TRUTH PostgreSQL; Qdrant yalnizca aranabilir turev. Bu modul
CAGIRANIN acik session'ini kullanir, commit ETMEZ (jobs_repository.enqueue
ile ayni sozlesme) - photo-scoped job handler'i (photo_service.
run_semantic_index_job) tek commit'te sonlandirir.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from qdrant_client.models import PointStruct
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ai.embedding import get_embedding_provider
from app.ai.embedding.base import EmbeddingProvider
from app.db import qdrant
from app.db.models import PhotoAnalysis

logger = logging.getLogger("photoai.semantic")


# --- JSON analiz -> metin dokumani -----------------------------------------


def _clean(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_list(values) -> list[str]:
    return [v.strip() for v in (values or []) if isinstance(v, str) and v.strip()]


def build_document(analysis: PhotoAnalysis) -> str:
    """photo_analysis satirini TEK metne cevirir. Turkce alan etiketleri
    bilincli: embedding modeline baglam verir ("Ruh hali: huzurlu" salt
    "huzurlu"dan daha ayirt edici). Onemli alanlar basta (description +
    ana nesne/eylem/ruh hali/kullanim). Bos/None alanlar satir olarak
    ATLANIR. `filename` BILEREK dahil edilmez - embedding'i kirletir,
    dosya adi eslesmesi zaten istemci fallback aramasinda var.
    """
    lines: list[str] = []

    desc = _clean(analysis.description)
    if desc:
        lines.append(desc)

    head: list[str] = []
    if _clean(analysis.primary_object):
        head.append(f"Ana nesne: {analysis.primary_object.strip()}")
    if _clean(analysis.action):
        head.append(f"Eylem: {analysis.action.strip()}")
    if _clean(analysis.mood):
        head.append(f"Ruh hali: {analysis.mood.strip()}")
    if _clean(analysis.use_case):
        head.append(f"Kullanim amaci: {analysis.use_case.strip()}")
    if head:
        lines.append(". ".join(head) + ".")

    for label, values in (
        ("Ikincil nesneler", analysis.secondary_objects),
        ("Ortam", analysis.environment),
        ("Oznitelikler", analysis.attributes),
        ("Baglam", analysis.context),
        ("Stil", analysis.style),
        ("Hedef kitle", analysis.audience),
    ):
        vals = _clean_list(values)
        if vals:
            lines.append(f"{label}: {', '.join(vals)}")

    pf_bits: list[str] = []
    for pf in (analysis.public_figures or []):
        if not isinstance(pf, dict):
            continue
        name = _clean(pf.get("name"))
        types = ", ".join(_clean_list(pf.get("types")))
        if name and types:
            pf_bits.append(f"{name} ({types})")
        elif name or types:
            pf_bits.append(name or types)
    if pf_bits:
        lines.append(f"Taninan kisiler: {'; '.join(pf_bits)}")

    tags = _clean_list(analysis.all_tags)
    if tags:
        lines.append(f"Etiketler: {', '.join(tags)}")

    return "\n".join(lines)


# --- Indeksleme ----------------------------------------------------------


_UPSERT_EMBEDDING_ROW = text(
    """
    INSERT INTO photo_embeddings (photo_id, model, dim, indexed_at)
    VALUES (:pid, :model, :dim, :ts)
    ON CONFLICT (photo_id) DO UPDATE
        SET model = EXCLUDED.model,
            dim = EXCLUDED.dim,
            indexed_at = EXCLUDED.indexed_at
    """
)


def index_documents(
    db: Session,
    provider: EmbeddingProvider,
    items: list[tuple[Any, str]],
) -> None:
    """(photo_id, dokuman) ciftlerini TOPLU embed edip Qdrant 'photo_semantic'
    koleksiyonuna + photo_embeddings izleme tablosuna yazar. Dokumanlar BOS
    OLMAMALI (cagiran suzer). COMMIT ETMEZ.

    Qdrant upsert'i PG yazimindan ONCE: cagiranin commit'i patlarsa Qdrant'ta
    sahipsiz nokta kalir - zararsiz, sonraki calismada uzerine yazilir (face
    pipeline'daki ayni bilinen sinir). Batching, yerel e5'in GPU verimliligi
    icin onemli (backfill'de binlerce dokuman) - bu yuzden ayri, cok-ogeli
    bir giris noktasi.
    """
    if not items:
        return
    vectors = provider.embed_documents([doc for _, doc in items])
    now = datetime.utcnow()
    now_iso = now.isoformat()
    qdrant.client.upsert(
        collection_name=qdrant.PHOTO_SEMANTIC_COLLECTION,
        points=[
            PointStruct(
                id=str(pid),
                vector=vec,
                payload={"photo_id": str(pid), "model": provider.name, "indexed_at": now_iso},
            )
            for (pid, _), vec in zip(items, vectors)
        ],
    )
    for pid, _ in items:
        db.execute(
            _UPSERT_EMBEDDING_ROW,
            {"pid": str(pid), "model": provider.name, "dim": provider.dim, "ts": now},
        )


def index_photo(db: Session, photo_id) -> bool:
    """photo_id'nin VLM analizinden bir embedding uretip Qdrant'a + izleme
    tablosuna yazar (tek fotograf - worker giris noktasi run_semantic_index_job
    bunu kullanir).

    Donus:
      False -> photo_analysis satiri HENUZ YOK (analiz bitmemis). Cagiran
               (run_semantic_index_job) bunu "cezasiz requeue" sinyali sayar.
      True  -> islendi (dokuman bos cikan dejenere analizde de True - is
               tamamlanmis sayilir, sonsuz requeue olmaz; sadece uyari loglanir).

    COMMIT ETMEZ.
    """
    analysis = (
        db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == photo_id).first()
    )
    if analysis is None:
        return False

    document = build_document(analysis)
    if not document.strip():
        logger.warning(
            "photo_id=%s icin VLM analizi var ama dokuman BOS cikti - "
            "embedding uretilmedi (dejenere analiz?).", photo_id,
        )
        return True

    index_documents(db, get_embedding_provider(), [(photo_id, document)])
    return True


def remove_from_index(db: Session, photo_id) -> None:
    """Fotograf silinirken cagrilir (photo_service.delete_photo). Qdrant
    noktasini siler; photo_embeddings satiri photos FK'si CASCADE ile zaten
    gider ama acik silme, delete_photo'nun kendi commit'inden ONCE
    calistiginda tutarli kalinmasini saglar."""
    qdrant.client.delete(
        collection_name=qdrant.PHOTO_SEMANTIC_COLLECTION,
        points_selector=[str(photo_id)],
    )
    db.execute(
        text("DELETE FROM photo_embeddings WHERE photo_id = :pid"),
        {"pid": str(photo_id)},
    )


# --- Arama -------------------------------------------------------------


def search(
    query: str, limit: int, min_score: float = 0.0
) -> list[tuple[str, float]]:
    """Sorguyu embed edip 'photo_semantic' uzerinde vektor benzerligiyle
    siralar. Donus: [(photo_id, score), ...] - skora gore azalan
    (Qdrant Cosine). Bos/whitespace sorgu -> bos liste.

    min_score: bu Cosine skorunun ALTINDA kalan adaylar Qdrant tarafinda
    elenir (score_threshold) - liste `limit`ten kisa, hatta bos donebilir.
    0.0 (varsayilan) = esik uygulanmaz. Cagiran genelde
    settings.SEMANTIC_SEARCH_MIN_SCORE gecirir.
    """
    q = query.strip()
    if not q:
        return []
    vector = get_embedding_provider().embed_query(q)
    result = qdrant.client.query_points(
        collection_name=qdrant.PHOTO_SEMANTIC_COLLECTION,
        query=vector,
        limit=limit,
        # 0.0'da da guvenle gecilebilir (Cosine'de tum skorlar >= -1),
        # ama None gecmek Qdrant'ta esigi tamamen atlar - niyet acik olsun.
        score_threshold=min_score if min_score > 0.0 else None,
    )
    return [(str(p.id), float(p.score)) for p in result.points]
