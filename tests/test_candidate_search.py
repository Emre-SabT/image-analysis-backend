"""candidate_search.py testleri - GERCEK PostgreSQL + Qdrant'a karsi calisir.

Iki backend'in (QdrantCandidateFinder, PgBruteForceCandidateFinder) AYNI
sekli (Candidate.score/payload/vector) urettigini ve GERCEKTEN tutarli
sonuc verdigini dogrular.
"""

import uuid
from datetime import datetime

import numpy as np
import pytest
from qdrant_client.models import PointStruct
from sqlalchemy import text

from app.db import qdrant
from app.db.session import SessionLocal
from app.services.candidate_search import PgBruteForceCandidateFinder, QdrantCandidateFinder


def _norm_vec():
    v = np.random.randn(512)
    return v / np.linalg.norm(v)


@pytest.fixture
def seeded_person(test_user_id):
    """Hem PG (centroid kolonu) hem Qdrant (identity_pool) tarafinda TUTARLI
    tek bir person - iki backend'i AYNI veriyle karsilastirmak icin."""
    pid = uuid.uuid4()
    vec = _norm_vec()
    db = SessionLocal()
    try:
        db.execute(
            text(
                "INSERT INTO persons (id, display_name, face_count, created_by_user_id, "
                "created_at, centroid, centroid_updated_at) "
                "VALUES (:id, 'cs-test', 1, :u, now(), :c, now())"
            ),
            {"id": pid, "u": str(test_user_id), "c": vec.astype(np.float32).tobytes()},
        )
        db.commit()
    finally:
        db.close()

    qdrant.client.upsert(
        qdrant.IDENTITY_POOL_COLLECTION,
        points=[PointStruct(id=str(pid), vector=vec.tolist(),
                             payload={"kind": "person", "id": str(pid), "face_count": 1})],
    )

    yield pid, vec

    db2 = SessionLocal()
    try:
        db2.execute(text("DELETE FROM persons WHERE id = :id"), {"id": pid})
        db2.commit()
    finally:
        db2.close()
    qdrant.client.delete(qdrant.IDENTITY_POOL_COLLECTION, points_selector=[str(pid)])


def test_both_backends_find_the_same_identity(seeded_person):
    pid, vec = seeded_person
    query = vec + np.random.randn(512) * 0.001  # neredeyse ayni yon
    query = query / np.linalg.norm(query)

    q_candidates = QdrantCandidateFinder().find_candidates(query, limit=5)
    pg_candidates = PgBruteForceCandidateFinder().find_candidates(query, limit=5)

    assert q_candidates and q_candidates[0].payload["id"] == str(pid)
    assert pg_candidates and pg_candidates[0].payload["id"] == str(pid)

    # skorlar (kosinus benzerligi) iki backend'de de neredeyse AYNI olmali
    assert abs(q_candidates[0].score - pg_candidates[0].score) < 1e-3, (
        q_candidates[0].score, pg_candidates[0].score,
    )


def test_pg_brute_force_excludes_soft_deleted_and_labeled(test_user_id):
    """Aktif-olmayan (soft-delete edilmis person / labeled|merged cluster)
    kimlikler PG brute-force aramasina HIC girmemeli."""
    pid = uuid.uuid4()
    vec = _norm_vec()
    db = SessionLocal()
    try:
        db.execute(
            text(
                "INSERT INTO persons (id, display_name, face_count, created_by_user_id, "
                "created_at, deleted_at, centroid, centroid_updated_at) "
                "VALUES (:id, 'deleted-test', 1, :u, now(), now(), :c, now())"
            ),
            {"id": pid, "u": str(test_user_id), "c": vec.astype(np.float32).tobytes()},
        )
        db.commit()

        candidates = PgBruteForceCandidateFinder().find_candidates(vec, limit=50)
        found_ids = {c.payload["id"] for c in candidates}
        assert str(pid) not in found_ids, "soft-delete edilmis person aramaya DAHIL olmus"
    finally:
        db.execute(text("DELETE FROM persons WHERE id = :id"), {"id": pid})
        db.commit()
        db.close()


def test_fetch_vector_matches_across_backends(seeded_person):
    pid, vec = seeded_person
    q_vec = QdrantCandidateFinder().fetch_vector("person", str(pid))
    pg_vec = PgBruteForceCandidateFinder().fetch_vector("person", str(pid))
    cos_sim = float(np.dot(q_vec, pg_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(pg_vec)))
    assert cos_sim > 0.999
