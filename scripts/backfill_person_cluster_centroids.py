r"""
PR-A (identity/centroid eszamanlilik calismasi) - persons.centroid /
clusters.centroid kolonlarini PG UYELIGINDEN (Qdrant identity_pool'dan
KOPYALAMA DEGIL) yeniden hesaplayip doldurur.

NEDEN QDRANT'TAN KOPYALAMA DEGIL, PG UYELIGINDEN YENIDEN HESAPLAMA:
Qdrant identity_pool'daki mevcut deger, gecmis artimli-guncelleme
yaris durumlarindan (bu calismanin konusu) etkilenmis, supheli olabilir.
"Temiz sayfa" ile baslamak - _upsert_identity_centroid'in (person_service.py)
zaten yaptigi isin PG hedefli hali - daha guvenilir bir baslangic noktasi.

NEDEN PARCALI (CHUNKED): 500K fotograf / ~1M yuz olcegindeki bir sahada
TUM embedding'leri tek seferde belleğe cekmek (982 yuz icin bile 221ms,
1M yuz icin dakikalar + ~2GB bellek gerektirir) kabul edilemez. Kimlikler
--chunk-size (varsayilan 500) buyuklugunde grup grup islenir; her grubun
SADECE o gruba ait yuzlerin embedding'i TEK Qdrant cagrisinda cekilir,
sonra bellekten dusurulur.

IS_BACKGROUND FILTRESI BILINCLI OLARAK UYGULANMIYOR: mevcut
person_service._upsert_identity_centroid (label_cluster/merge_identities/
reassign_face/delete_photo'nun kullandigi RECOMPUTE fonksiyonu) de bu
filtreyi uygulamiyor (bilinen, ayri bir bug - bkz. gorev: "Recompute
fonksiyonlari is_background yuzleri centroid'e dahil ediyor"). Bu script
MEVCUT davranisla TUTARLI kalmak icin BILEREK ayni sekilde davraniyor -
aksi halde backfill'in urettigi deger, ayni kimlik uzerinde bir sonraki
label_cluster/merge cagrisinin uretecegi degerden FARKLI olur, yeni bir
tutarsizlik yaratirdi. is_background duzeltmesi ayri bir gorev.

IDEMPOTENT: duz bir UPDATE (upsert degil, kolonlar zaten var) - kesintiye
ugrarsa BASTAN calistirmak GUVENLIDIR, veri kaybi riski yoktur (sadece
zaman kaybi). --resume-after-person / --resume-after-cluster ile id
sirasina gore kaldigi yerden devam da edilebilir (buyuk sahalarda script
kesintiye ugrarsa bastan baslamamak icin).

Kullanim:
    python scripts/backfill_person_cluster_centroids.py
    python scripts/backfill_person_cluster_centroids.py --chunk-size 200
    python scripts/backfill_person_cluster_centroids.py --resume-after-person <uuid>
"""

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import text

from app.db import qdrant
from app.db.session import SessionLocal


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _fetch_member_face_ids(db, table: str, id_column: str, chunk_ids: list[str]) -> dict[str, list[str]]:
    """Bir kimlik grubunun UYELIK bilgisini (hangi Face hangi kimlige ait)
    TEK sorguda ceker - PG kaynak-of-truth, Qdrant burada hic sorulmuyor."""
    rows = db.execute(
        text(f"SELECT id, {id_column} FROM faces WHERE {id_column}::text = ANY(:ids)"),
        {"ids": chunk_ids},
    ).fetchall()
    by_identity: dict[str, list[str]] = {i: [] for i in chunk_ids}
    for face_id, identity_id in rows:
        by_identity[str(identity_id)].append(str(face_id))
    return by_identity


def _backfill_chunk(db, table: str, id_column: str, chunk_ids: list[str]) -> tuple[int, int]:
    """Bir grup kimlik icin: uyelik PG'den, embedding'ler Qdrant'tan (TEK
    toplu cagri) cekilir, ortalama hesaplanir, centroid kolonuna yazilir.
    Donus: (guncellenen, atlanan-uyesiz) sayisi."""
    members = _fetch_member_face_ids(db, table, id_column, chunk_ids)
    all_face_ids = [fid for fids in members.values() for fid in fids]

    if not all_face_ids:
        return 0, len(chunk_ids)

    records = qdrant.client.retrieve(
        collection_name=qdrant.FACES_COLLECTION, ids=all_face_ids, with_vectors=True
    )
    vec_by_face_id = {r.id: np.asarray(r.vector, dtype=np.float32) for r in records}

    updated = skipped = 0
    for identity_id, face_ids in members.items():
        vectors = [vec_by_face_id[fid] for fid in face_ids if fid in vec_by_face_id]
        if not vectors:
            skipped += 1
            continue
        centroid = np.mean(vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        db.execute(
            text(f"UPDATE {table} SET centroid = :c WHERE id = :i"),
            {"c": centroid.astype(np.float32).tobytes(), "i": identity_id},
        )
        updated += 1
    db.commit()
    return updated, skipped


def backfill_persons(db, chunk_size: int, resume_after: str | None) -> None:
    query = "SELECT id::text FROM persons WHERE deleted_at IS NULL"
    params = {}
    if resume_after:
        query += " AND id::text > :after"
        params["after"] = resume_after
    query += " ORDER BY id"
    ids = [r[0] for r in db.execute(text(query), params).fetchall()]
    print(f"[person] islenecek aktif kimlik: {len(ids)}")

    total_updated = total_skipped = 0
    for i, chunk in enumerate(_chunked(ids, chunk_size)):
        updated, skipped = _backfill_chunk(db, "persons", "person_id", chunk)
        total_updated += updated
        total_skipped += skipped
        print(f"[person] grup {i + 1}: {len(chunk)} kimlik, {updated} guncellendi, "
              f"{skipped} atlandi (uyesiz) - son id={chunk[-1]}")
    print(f"[person] TOPLAM: {total_updated} guncellendi, {total_skipped} atlandi")


def backfill_clusters(db, chunk_size: int, resume_after: str | None) -> None:
    # SADECE unlabeled kumeler - labeled/merged olanlar zaten emekli
    # (bkz. label_cluster/merge_identities'in Asama-0 temizligi), onlara
    # centroid yazmanin bir anlami yok.
    query = "SELECT id::text FROM clusters WHERE status = 'unlabeled'"
    params = {}
    if resume_after:
        query += " AND id::text > :after"
        params["after"] = resume_after
    query += " ORDER BY id"
    ids = [r[0] for r in db.execute(text(query), params).fetchall()]
    print(f"[cluster] islenecek unlabeled kimlik: {len(ids)}")

    total_updated = total_skipped = 0
    for i, chunk in enumerate(_chunked(ids, chunk_size)):
        updated, skipped = _backfill_chunk(db, "clusters", "cluster_id", chunk)
        total_updated += updated
        total_skipped += skipped
        print(f"[cluster] grup {i + 1}: {len(chunk)} kimlik, {updated} guncellendi, "
              f"{skipped} atlandi (uyesiz) - son id={chunk[-1]}")
    print(f"[cluster] TOPLAM: {total_updated} guncellendi, {total_skipped} atlandi")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="persons/clusters.centroid'i PG uyelikten (Qdrant FACES_COLLECTION "
                     "embedding'leriyle) yeniden hesaplayip doldurur - parcali, idempotent."
    )
    parser.add_argument("--chunk-size", type=int, default=500,
                         help="Bir grupta islenecek kimlik sayisi (varsayilan 500 - "
                              "bellek/istek boyutunu sinirlar, kimlik basina ortalama "
                              "uye sayisina gore ayarlayin)")
    parser.add_argument("--resume-after-person", type=str, default=None,
                         help="Bu UUID'den SONRAKI person'lardan devam et (kesintiye "
                              "ugramis buyuk-olcek calistirmayi yeniden baslatmak icin)")
    parser.add_argument("--resume-after-cluster", type=str, default=None,
                         help="Bu UUID'den SONRAKI cluster'lardan devam et")
    parser.add_argument("--skip-persons", action="store_true")
    parser.add_argument("--skip-clusters", action="store_true")
    args = parser.parse_args()

    if args.resume_after_person:
        uuid.UUID(args.resume_after_person)  # erken dogrulama
    if args.resume_after_cluster:
        uuid.UUID(args.resume_after_cluster)

    db = SessionLocal()
    try:
        if not args.skip_persons:
            backfill_persons(db, args.chunk_size, args.resume_after_person)
        if not args.skip_clusters:
            backfill_clusters(db, args.chunk_size, args.resume_after_cluster)
    finally:
        db.close()

    print("\nBackfill tamamlandi.")


if __name__ == "__main__":
    main()
