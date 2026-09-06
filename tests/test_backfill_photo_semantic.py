"""scripts/backfill_photo_semantic.py testleri - GERCEK PostgreSQL + Qdrant,
deterministik sahte embedding provider (torch/AWS gerekmez).
"""

import uuid

import numpy as np
import pytest
from sqlalchemy import text

import app.ai.embedding as embedding_pkg
from app.core.settings import settings
from app.db import qdrant
from scripts.backfill_photo_semantic import _candidate_ids, _process_chunk


@pytest.fixture
def fake_provider():
    class _Fake:
        name = "fake:backfill-e5"
        dim = settings.EMBEDDING_DIM

        def _vec(self, s: str):
            rng = np.random.default_rng(abs(hash(s)) % (2**32))
            v = rng.standard_normal(self.dim)
            return (v / np.linalg.norm(v)).tolist()

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    prev = embedding_pkg._provider
    embedding_pkg._provider = _Fake()
    yield embedding_pkg._provider
    embedding_pkg._provider = prev


@pytest.fixture
def archive(db_session, fake_provider):
    """3 fotograf:
      a -> analizli, indekssiz          (backfill'e girmeli)
      b -> analizli, GUNCEL modelle indeksli   (girmemeli)
      c -> analizli, ESKI modelle indeksli     (girmeli)
    ayrica d -> analizi TAMAMEN bos (dokuman bos -> atlanmali).
    """
    from app.db.qdrant import ensure_collections

    ensure_collections()
    ids = {k: uuid.uuid4() for k in ("a", "b", "c", "d")}
    for key, pid in ids.items():
        db_session.execute(
            text(
                "INSERT INTO photos (id, filename, storage_path, status, created_at) "
                "VALUES (:id, :fn, '/tmp/x.jpg', 'analyzed', now())"
            ),
            {"id": pid, "fn": f"bf-{key}.jpg"},
        )
    db_session.execute(
        text("INSERT INTO photo_analysis (photo_id, description, all_tags) "
             "VALUES (:id, :d, :t)"),
        {"id": ids["a"], "d": "kirmizi bisiklet parkta", "t": ["bisiklet", "park"]},
    )
    db_session.execute(
        text("INSERT INTO photo_analysis (photo_id, description) VALUES (:id, :d)"),
        {"id": ids["b"], "d": "mavi kapi"},
    )
    db_session.execute(
        text("INSERT INTO photo_analysis (photo_id, description) VALUES (:id, :d)"),
        {"id": ids["c"], "d": "sari duvar"},
    )
    db_session.execute(
        text("INSERT INTO photo_analysis (photo_id) VALUES (:id)"),  # tum alanlar bos
        {"id": ids["d"]},
    )
    # b: guncel modelle, c: eski modelle indekslenmis say
    db_session.execute(
        text("INSERT INTO photo_embeddings (photo_id, model, dim, indexed_at) "
             "VALUES (:id, :m, :dim, now())"),
        {"id": ids["b"], "m": "fake:backfill-e5", "dim": settings.EMBEDDING_DIM},
    )
    db_session.execute(
        text("INSERT INTO photo_embeddings (photo_id, model, dim, indexed_at) "
             "VALUES (:id, :m, :dim, now())"),
        {"id": ids["c"], "m": "eski-model-v1", "dim": settings.EMBEDDING_DIM},
    )
    db_session.commit()
    yield ids, db_session
    for pid in ids.values():
        db_session.execute(text("DELETE FROM photo_embeddings WHERE photo_id = :id"), {"id": pid})
        db_session.execute(text("DELETE FROM photo_analysis WHERE photo_id = :id"), {"id": pid})
        db_session.execute(text("DELETE FROM photos WHERE id = :id"), {"id": pid})
        qdrant.client.delete(
            collection_name=qdrant.PHOTO_SEMANTIC_COLLECTION, points_selector=[str(pid)]
        )
    db_session.commit()


def test_candidate_ids_selects_unindexed_and_stale_model(archive):
    ids, db = archive
    got = set(_candidate_ids(db, "fake:backfill-e5", reindex_all=False, resume_after=None))
    assert str(ids["a"]) in got  # indekssiz
    assert str(ids["c"]) in got  # eski model
    assert str(ids["d"]) in got  # bos dokuman ama analiz var -> aday; chunk isleme atlar
    assert str(ids["b"]) not in got  # guncel model


def test_candidate_ids_reindex_all_includes_everything(archive):
    ids, db = archive
    got = set(_candidate_ids(db, "fake:backfill-e5", reindex_all=True, resume_after=None))
    assert {str(v) for v in ids.values()} <= got


def test_candidate_ids_resume_after(archive):
    ids, db = archive
    all_sorted = sorted(str(v) for v in ids.values())
    pivot = all_sorted[1]
    got = _candidate_ids(db, "fake:backfill-e5", reindex_all=True, resume_after=pivot)
    assert all(x > pivot for x in got)


def test_process_chunk_indexes_and_skips_empty(archive):
    ids, db = archive
    provider = embedding_pkg.get_embedding_provider()
    indexed, empty = _process_chunk(db, provider, [str(ids["a"]), str(ids["d"])])

    assert (indexed, empty) == (1, 1)
    row = db.execute(
        text("SELECT model FROM photo_embeddings WHERE photo_id = :id"), {"id": ids["a"]}
    ).first()
    assert row is not None and row.model == "fake:backfill-e5"

    # Qdrant'a nokta yazildi mi - dogrudan retrieve (search siralamasina
    # bagli DEGIL: 'photo_semantic' koleksiyonu gercek backfill'den 463+
    # nokta icerebilir, top-N icinde olma garantisi yok).
    got = {str(p.id) for p in qdrant.client.retrieve(
        collection_name=qdrant.PHOTO_SEMANTIC_COLLECTION,
        ids=[str(ids["a"]), str(ids["d"])],
    )}
    assert str(ids["a"]) in got   # indekslendi
    assert str(ids["d"]) not in got  # bos dokuman -> atlandi
