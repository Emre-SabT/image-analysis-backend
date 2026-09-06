"""semantic_service.py testleri.

- build_document: saf, DB'siz - Turkce etiketleme, bos alan atlama, public_figures.
- index_photo / search / remove_from_index: GERCEK PostgreSQL + Qdrant'a karsi,
  embedding modeli yerine deterministik SAHTE provider (torch/AWS gerekmez).
"""

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import text

import app.ai.embedding as embedding_pkg
from app.core.settings import settings
from app.db import qdrant
from app.db.models import PhotoAnalysis
from app.services import semantic_service


# --- build_document (saf) ------------------------------------------------


def _analysis(**over):
    base = dict(
        description=None, primary_object=None, action=None, mood=None, use_case=None,
        secondary_objects=[], environment=[], attributes=[], context=[], style=[],
        audience=[], public_figures=[], all_tags=[],
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_build_document_full_has_description_first_and_labeled_head():
    doc = semantic_service.build_document(_analysis(
        description="Deniz kenarinda gun batimi.",
        primary_object="sahil", action="yuruyus", mood="huzurlu", use_case="seyahat blogu",
        all_tags=["deniz", "gun batimi"],
    ))
    lines = doc.split("\n")
    assert lines[0] == "Deniz kenarinda gun batimi."
    assert lines[1] == "Ana nesne: sahil. Eylem: yuruyus. Ruh hali: huzurlu. Kullanim amaci: seyahat blogu."
    assert "Etiketler: deniz, gun batimi" in lines


def test_build_document_skips_empty_fields():
    doc = semantic_service.build_document(_analysis(description="Sadece aciklama"))
    assert doc == "Sadece aciklama"
    assert "Ana nesne" not in doc
    assert "Etiketler" not in doc


def test_build_document_all_empty_returns_blank():
    assert semantic_service.build_document(_analysis()) == ""


def test_build_document_none_arrays_do_not_crash():
    doc = semantic_service.build_document(_analysis(
        description="x", environment=None, all_tags=None, public_figures=None,
    ))
    assert doc == "x"


def test_build_document_public_figures_formatting():
    doc = semantic_service.build_document(_analysis(
        public_figures=[
            {"name": "Ali Veli", "types": ["sporcu", "aktivist"]},
            {"name": "", "types": ["sanatci"]},
            {"name": "Ayse", "types": []},
        ],
    ))
    assert "Taninan kisiler: Ali Veli (sporcu, aktivist); sanatci; Ayse" in doc


# --- index_photo / search / remove (entegrasyon) -----------------------


@pytest.fixture
def fake_provider():
    """Deterministik: ayni metin -> ayni birim-norm vektor. embed_query ve
    embed_documents AYNI kaynagi kullanir, boylece indekslenen dokuman
    stringiyle aranınca skor ~1.0 olur."""
    class _Fake:
        name = "fake:test-e5"
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
def photo_with_analysis(db_session, fake_provider):
    from app.db.qdrant import ensure_collections

    ensure_collections()  # 'photo_semantic' koleksiyonu (dim=EMBEDDING_DIM)
    pid = uuid.uuid4()
    db_session.execute(
        text(
            "INSERT INTO photos (id, filename, storage_path, status, created_at) "
            "VALUES (:id, 'sem-test.jpg', '/tmp/sem-test.jpg', 'analyzed', now())"
        ),
        {"id": pid},
    )
    db_session.execute(
        text(
            "INSERT INTO photo_analysis (photo_id, description, primary_object, all_tags) "
            "VALUES (:id, :d, :po, :tags)"
        ),
        {"id": pid, "d": "Deniz kenarinda gun batimi", "po": "sahil",
         "tags": ["deniz", "gun batimi", "sahil"]},
    )
    db_session.commit()
    yield pid
    db_session.execute(text("DELETE FROM photo_embeddings WHERE photo_id = :id"), {"id": pid})
    db_session.execute(text("DELETE FROM photo_analysis WHERE photo_id = :id"), {"id": pid})
    db_session.execute(text("DELETE FROM photos WHERE id = :id"), {"id": pid})
    db_session.commit()
    qdrant.client.delete(
        collection_name=qdrant.PHOTO_SEMANTIC_COLLECTION, points_selector=[str(pid)]
    )


def test_index_photo_false_when_no_analysis(db_session, fake_provider):
    pid = uuid.uuid4()
    db_session.execute(
        text(
            "INSERT INTO photos (id, filename, storage_path, status, created_at) "
            "VALUES (:id, 'no-analysis.jpg', '/tmp/x.jpg', 'processing', now())"
        ),
        {"id": pid},
    )
    db_session.commit()
    try:
        assert semantic_service.index_photo(db_session, pid) is False
    finally:
        db_session.execute(text("DELETE FROM photos WHERE id = :id"), {"id": pid})
        db_session.commit()


def test_index_and_search_roundtrip(db_session, photo_with_analysis):
    pid = photo_with_analysis

    assert semantic_service.index_photo(db_session, pid) is True
    db_session.commit()

    row = db_session.execute(
        text("SELECT model, dim FROM photo_embeddings WHERE photo_id = :id"), {"id": pid}
    ).first()
    assert row is not None
    assert row.model == "fake:test-e5"
    assert row.dim == settings.EMBEDDING_DIM

    analysis = db_session.query(PhotoAnalysis).filter_by(photo_id=pid).first()
    doc = semantic_service.build_document(analysis)

    hits = semantic_service.search(doc, limit=10)
    assert hits, "arama hic sonuc dondurmedi"
    assert hits[0][0] == str(pid)
    assert hits[0][1] > 0.99  # ayni vektor -> kosinus ~1


def test_remove_from_index(db_session, photo_with_analysis):
    pid = photo_with_analysis
    semantic_service.index_photo(db_session, pid)
    db_session.commit()

    analysis = db_session.query(PhotoAnalysis).filter_by(photo_id=pid).first()
    doc = semantic_service.build_document(analysis)
    assert any(h[0] == str(pid) for h in semantic_service.search(doc, limit=10))

    semantic_service.remove_from_index(db_session, pid)
    db_session.commit()

    assert not any(h[0] == str(pid) for h in semantic_service.search(doc, limit=10))
    gone = db_session.execute(
        text("SELECT 1 FROM photo_embeddings WHERE photo_id = :id"), {"id": pid}
    ).first()
    assert gone is None


def test_search_blank_query_returns_empty(fake_provider):
    assert semantic_service.search("   ", limit=10) == []


# --- run_semantic_index_job (worker giris noktasi) --------------------


def test_run_semantic_index_job_indexes(db_session, photo_with_analysis):
    from app.services import photo_service

    pid = photo_with_analysis
    photo_service.run_semantic_index_job(pid)  # kendi SessionLocal'ini acar

    row = db_session.execute(
        text("SELECT model FROM photo_embeddings WHERE photo_id = :id"), {"id": pid}
    ).first()
    assert row is not None and row.model == "fake:test-e5"

    analysis = db_session.query(PhotoAnalysis).filter_by(photo_id=pid).first()
    hits = semantic_service.search(semantic_service.build_document(analysis), limit=10)
    assert hits and hits[0][0] == str(pid)


def test_delete_photo_removes_semantic_index(db_session, photo_with_analysis):
    from app.services import photo_service

    pid = photo_with_analysis
    photo_service.run_semantic_index_job(pid)

    analysis = db_session.query(PhotoAnalysis).filter_by(photo_id=pid).first()
    doc = semantic_service.build_document(analysis)
    assert any(h[0] == str(pid) for h in semantic_service.search(doc, limit=10))

    photo_service.delete_photo(db_session, pid, None)

    assert not any(h[0] == str(pid) for h in semantic_service.search(doc, limit=10))
    assert db_session.execute(
        text("SELECT 1 FROM photo_embeddings WHERE photo_id = :id"), {"id": pid}
    ).first() is None
    db_session.execute(text("DELETE FROM activity_log WHERE target_id = :id"), {"id": pid})
    db_session.commit()


def test_run_semantic_index_job_lockconflict_when_analysis_missing(db_session, fake_provider):
    from app.services.photo_service import LockConflict, run_semantic_index_job

    pid = uuid.uuid4()
    db_session.execute(
        text(
            "INSERT INTO photos (id, filename, storage_path, status, created_at) "
            "VALUES (:id, 'pending.jpg', '/tmp/pending.jpg', 'processing', now())"
        ),
        {"id": pid},
    )
    db_session.commit()
    try:
        with pytest.raises(LockConflict):
            run_semantic_index_job(pid)
    finally:
        db_session.execute(text("DELETE FROM photos WHERE id = :id"), {"id": pid})
        db_session.commit()
