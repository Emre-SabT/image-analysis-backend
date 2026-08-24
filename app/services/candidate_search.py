"""Kimlik-havuzu (person+cluster) aday arama SOYUTLAMASI (PR-B).

Amac: _assign_or_bucket'in (face_service.py) hangi arama motorunu (Qdrant
HNSW ayna / PG brute-force) kullandigini bilmeden calismasini saglamak -
olcek kararini ("centroid PG'de mi Qdrant'ta mi aranir" tartismasi, bkz.
konusma gecmisi) bu TEK noktaya erteler. Centroid PG'de OTORITER (PR-A/
PR-C - persons.centroid / clusters.centroid); Qdrant identity_pool GECIS
DONEMI boyunca dual-write ile senkron tutuluyor (bkz. face_service.py:
_update_identity_centroid_locked).

KARAR (bu tur): PgBruteForceCandidateFinder ONBELLEKSIZ - her aramada
dogrudan PG'den taze BYTEA fetch. Gerekce: iki-surecli gercek olcumde
(a) hafif diff-sorgusu vs (c) onbellek-yok arasindaki fark (%0 vs %1.2
bayat, 40 iterasyonluk TEK seed) onbellegin getirdigi karmasikligi
(senkron durumu, invalidation mantigi, surec-ici state) haklı cikaracak
kadar guclu degil. Arama bayatligi zaten bir KALITE sorunu - veri
butunlugu SELECT FOR UPDATE ile AYRICA korunuyor (bkz.
face_service._update_identity_centroid_locked), onbellek bunu degistirmez.
Bugunku olcekte BYTEA fetch (olculdu: 2.38ms, N=264) zaten Qdrant'in
query_points'inden (~16.75ms) hizli. Gercek bir darbogaz OLCULURSE
onbellek BURAYA, tek yere eklenir - PgBruteForceCandidateFinder'in
disina hicbir sey sizmaz (find_candidates imzasi degismez).

Not: N cok buyudugunde (binlerce+ kimlik) dogru cevap "brute-force'a
onbellek eklemek" olmayabilir - QdrantCandidateFinder zaten var ve
HNSW ile sub-linear olcekleniyor; o noktada backend'i geri Qdrant'a
cevirmek (tek ayar) muhtemelen onbellek insa etmekten daha basit.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sqlalchemy import text

from app.core.settings import settings
from app.db import qdrant
from app.db.session import SessionLocal


@dataclass
class Candidate:
    """qdrant_client.models.ScoredPoint'in _assign_or_bucket'in KULLANDIGI
    alt kumesi - iki backend'in de doldurabildigi ORTAK sekil. score:
    kosinus benzerligi (Qdrant ayni metrigi kullaniyor, PG brute-force
    normalize vektorlerin ic carpimiyla ayni sonucu uretir)."""

    score: float
    payload: dict
    vector: list[float] | None = None


class CandidateFinder(Protocol):
    def find_candidates(self, embedding: np.ndarray, limit: int) -> list[Candidate]: ...
    def fetch_vector(self, kind: str, identity_id: str) -> np.ndarray: ...


class QdrantCandidateFinder:
    """Bugunku davranisi SARAN ayna - hicbir mantik degismedi, sadece
    donus tipi Candidate'e cevriliyor."""

    def find_candidates(self, embedding: np.ndarray, limit: int) -> list[Candidate]:
        result = qdrant.client.query_points(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION,
            query=embedding.tolist(),
            limit=limit,
            with_vectors=True,
        )
        return [
            Candidate(score=p.score, payload=p.payload or {}, vector=p.vector)
            for p in result.points
        ]

    def fetch_vector(self, kind: str, identity_id: str) -> np.ndarray:
        records = qdrant.client.retrieve(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION,
            ids=[identity_id],
            with_vectors=True,
        )
        if not records:
            raise RuntimeError(f"Kimlik merkezi Qdrant'ta bulunamadi: kind={kind} id={identity_id}")
        return np.asarray(records[0].vector, dtype=np.float64)


_BRUTE_FORCE_SQL = """
    SELECT id, 'person' AS kind, centroid, face_count AS member_count
    FROM persons WHERE deleted_at IS NULL AND centroid IS NOT NULL
    UNION ALL
    SELECT id, 'cluster' AS kind, centroid, size AS member_count
    FROM clusters WHERE status = 'unlabeled' AND centroid IS NOT NULL
"""


class PgBruteForceCandidateFinder:
    """PR-C sonrasi otoriter kaynaktan (persons/clusters.centroid) brute-
    force arama. ONBELLEKSIZ (bkz. modul basi karar notu) - her cagrida
    taze fetch, N buyudukce maliyeti (olculmus) dogrusala-yakin artar:
    N=264 -> 2.38ms, N=1000 -> 21.45ms, N=5000 -> 108.79ms, N=20000 ->
    746.15ms (BYTEA + np.frombuffer, tum degerler olculdu)."""

    def find_candidates(self, embedding: np.ndarray, limit: int) -> list[Candidate]:
        db = SessionLocal()
        try:
            rows = db.execute(text(_BRUTE_FORCE_SQL)).fetchall()
        finally:
            db.close()
        if not rows:
            return []

        vectors = np.stack(
            [np.frombuffer(bytes(r.centroid), dtype=np.float32).astype(np.float64) for r in rows]
        )
        q = embedding.astype(np.float64)
        q = q / np.linalg.norm(q)
        sims = vectors @ q  # centroid'ler zaten normalize (backfill + write yolu garantisi)

        order = np.argsort(-sims)[:limit]
        return [
            Candidate(
                score=float(sims[i]),
                payload={
                    "kind": rows[i].kind,
                    "id": str(rows[i].id),
                    "face_count": rows[i].member_count,
                },
                vector=vectors[i].tolist(),
            )
            for i in order
        ]

    def fetch_vector(self, kind: str, identity_id: str) -> np.ndarray:
        table = "persons" if kind == "person" else "clusters"
        db = SessionLocal()
        try:
            row = db.execute(
                text(f"SELECT centroid FROM {table} WHERE id = :id"),
                {"id": uuid.UUID(identity_id)},
            ).fetchone()
        finally:
            db.close()
        if row is None or row[0] is None:
            raise RuntimeError(f"Kimlik merkezi PG'de bulunamadi: kind={kind} id={identity_id}")
        return np.frombuffer(bytes(row[0]), dtype=np.float32).astype(np.float64)


def get_candidate_finder() -> CandidateFinder:
    """Aktif arama backend'i - settings.IDENTITY_SEARCH_BACKEND ile secilir.
    Cutover: worker durdurmadan, TEK ayar degisikligiyle cevrilir (bkz.
    face_service.py basindaki cutover notu). Varsayilan 'qdrant' - PR-C
    deploy edildiginde bile davranis DEGISMEZ, sadece dual-write baslar;
    okuma tarafi ayri, bilincli bir adimda 'pg_brute_force'a cevrilir."""
    if settings.IDENTITY_SEARCH_BACKEND == "pg_brute_force":
        return PgBruteForceCandidateFinder()
    return QdrantCandidateFinder()
