"""Madde 4: identity_locks.IdentityLockTimeout, photo_service.
run_face_pipeline_job icinde LockConflict'e cevriliyor mu - GERCEK DB'ye
karsi, ama detect_and_embed'in kendisini monkeypatch'leyerek (bu testin
amaci ceviri noktasi, tespit/embedding degil).
"""

import uuid

import pytest
from sqlalchemy import text

from app.db import identity_locks
from app.db.models import Photo
from app.db.session import SessionLocal
from app.services import face_service, photo_service
from app.services.photo_service import LockConflict


@pytest.fixture
def photo(test_user_id):
    db = SessionLocal()
    try:
        p = Photo(filename="t.jpg", storage_path=f"uploads/{uuid.uuid4()}.jpg",
                   uploaded_by_user_id=test_user_id)
        db.add(p)
        db.commit()
        db.refresh(p)
        pid = p.id
    finally:
        db.close()

    yield pid

    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(pid)})
        db.commit()
    finally:
        db.close()


def test_identity_lock_timeout_becomes_lock_conflict(monkeypatch, photo):
    def raise_identity_timeout(db, photo_obj):
        raise identity_locks.IdentityLockTimeout("simulated: baska islem kimligi tutuyor")

    monkeypatch.setattr(face_service, "detect_and_embed", raise_identity_timeout)

    with pytest.raises(LockConflict):
        photo_service.run_face_pipeline_job(photo)
