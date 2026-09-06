r"""VLM analizi (photo_analysis) OLAN ama semantik arama indeksinde
(photo_embeddings + Qdrant 'photo_semantic') OLMAYAN - ya da FARKLI bir
modelle indekslenmis - fotograflari toplu indeksler.

NE ZAMAN CALISTIRILIR:
  - Ozellik ilk devreye alinirken (mevcut arsivi bir kez indekslemek).
  - Embedding provider/model degistiginde (EMBEDDING_PROVIDER veya
    EMBEDDING_DIM): once Qdrant 'photo_semantic' koleksiyonunu DUSURUN
    (boyut degistiyse ensure_collections zaten reddeder), sonra bu scripti
    --reindex-all ile calistirin.

PARCALI (CHUNKED): --chunk-size (varsayilan 100) buyuklugunde gruplar;
her grup TEK embed cagrisinda (yerel e5 icin GPU batch verimliligi) islenir.

IDEMPOTENT: index_documents hem Qdrant upsert hem photo_embeddings
ON CONFLICT kullanir - kesintiye ugrarsa BASTAN calistirmak GUVENLIDIR.
--resume-after <uuid> ile photo_id sirasina gore kaldigi yerden devam
edilebilir.

Kullanim:
    python scripts/backfill_photo_semantic.py
    python scripts/backfill_photo_semantic.py --chunk-size 200
    python scripts/backfill_photo_semantic.py --reindex-all
    python scripts/backfill_photo_semantic.py --resume-after <uuid>
"""

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.ai.embedding import get_embedding_provider
from app.db import qdrant
from app.db.models import PhotoAnalysis
from app.db.qdrant import ensure_collections
from app.db.session import SessionLocal
from app.services import semantic_service


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _candidate_ids(db, model_name: str, reindex_all: bool, resume_after: str | None) -> list[str]:
    """VLM analizi olan ama (indekslenmemis / farkli modelle indekslenmis)
    fotograf id'leri - photo_id sirasinda."""
    query = """
        SELECT pa.photo_id::text
        FROM photo_analysis pa
        LEFT JOIN photo_embeddings pe ON pe.photo_id = pa.photo_id
        WHERE 1 = 1
    """
    params: dict = {}
    if not reindex_all:
        query += " AND (pe.photo_id IS NULL OR pe.model <> :model)"
        params["model"] = model_name
    if resume_after:
        query += " AND pa.photo_id::text > :after"
        params["after"] = resume_after
    query += " ORDER BY pa.photo_id"
    return [r[0] for r in db.execute(text(query), params).fetchall()]


def _process_chunk(db, provider, chunk_ids: list[str]) -> tuple[int, int]:
    """Bir grup: analiz satirlari cekilir, dokumanlar kurulur, BOS olmayanlar
    toplu embed edilip yazilir. Donus: (indekslenen, atlanan-bos-dokuman)."""
    chunk_uuids = [uuid.UUID(i) for i in chunk_ids]
    rows = (
        db.query(PhotoAnalysis)
        .filter(PhotoAnalysis.photo_id.in_(chunk_uuids))
        .all()
    )
    items: list[tuple[uuid.UUID, str]] = []
    empty = 0
    for r in rows:
        doc = semantic_service.build_document(r)
        if doc.strip():
            items.append((r.photo_id, doc))
        else:
            empty += 1

    semantic_service.index_documents(db, provider, items)
    db.commit()
    return len(items), empty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="photo_analysis olan fotograflari semantik arama indeksine "
                    "(Qdrant 'photo_semantic' + photo_embeddings) toplu yazar - "
                    "parcali, idempotent."
    )
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Bir embed cagrisinda islenecek dokuman sayisi (varsayilan 100)")
    parser.add_argument("--reindex-all", action="store_true",
                        help="Zaten indekslenmis olanlari da yeniden isle "
                             "(provider/model degisiminde)")
    parser.add_argument("--resume-after", type=str, default=None,
                        help="Bu UUID'den SONRAKI photo_id'lerden devam et")
    args = parser.parse_args()

    if args.resume_after:
        uuid.UUID(args.resume_after)  # erken dogrulama
    if args.chunk_size < 1:
        parser.error("--chunk-size en az 1 olmali")

    # Koleksiyonu sagla (boyut uyusmazliginda ensure_collections RuntimeError
    # firlatir - "koleksiyonu dusurup yeniden calistirin" mesajiyla).
    ensure_collections()

    provider = get_embedding_provider()
    print(f"Embedding provider: {provider.name} (dim={provider.dim})")

    db = SessionLocal()
    try:
        ids = _candidate_ids(db, provider.name, args.reindex_all, args.resume_after)
        print(f"Indekslenecek fotograf: {len(ids)}"
              + (" (--reindex-all)" if args.reindex_all else ""))

        total_indexed = total_empty = 0
        for i, chunk in enumerate(_chunked(ids, args.chunk_size)):
            indexed, empty = _process_chunk(db, provider, chunk)
            total_indexed += indexed
            total_empty += empty
            print(f"  grup {i + 1}: {len(chunk)} foto, {indexed} indekslendi, "
                  f"{empty} atlandi (bos dokuman) - son id={chunk[-1]}")

        print(f"\nTOPLAM: {total_indexed} indekslendi, {total_empty} atlandi (bos dokuman).")
    finally:
        db.close()

    print("Backfill tamamlandi.")


if __name__ == "__main__":
    main()
