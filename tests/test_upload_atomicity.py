"""POST /photos'un atomikligi: foto + 2 job TEK transaction'da.

Bu testin gecebilmesi icin photo_service.save_upload'in commit() DEGIL
flush() yapmasi gerekir. Onceki tasarimda save_upload kendi icinde commit
ediyordu; o durumda job insert patlasa bile foto satiri kalici olurdu ve
sistemde ASLA islenmeyecek "oksuz" fotograflar birikirdi.
"""

import io
import uuid

import pytest
from fastapi import UploadFile
from sqlalchemy import text

from app.db import jobs_repository as jr
from app.db.models import JOB_TYPE_FACE_PIPELINE, JOB_TYPE_VLM_ANALYSIS, Photo
from app.db.session import SessionLocal
from app.services import photo_service


def _upload_file(name="atomicity-test.jpg"):
    # 1x1 gecerli JPEG - save_upload sadece bayt okur, cozmez.
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), "red").save(buf, format="JPEG")
    buf.seek(0)
    return UploadFile(filename=name, file=buf)


def _cleanup(photo_id, user_id):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM jobs WHERE user_id=:u"), {"u": str(user_id)})
        db.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(photo_id)})
        db.commit()
    finally:
        db.close()


def test_save_upload_does_not_commit_on_its_own(test_user_id):
    """save_upload flush yapmali, commit ETMEMELI."""
    db = SessionLocal()
    photo_id = None
    try:
        photo, dup = photo_service.save_upload(db, _upload_file(f"a-{uuid.uuid4()}.jpg"),
                                               test_user_id)
        photo_id = photo.id
        assert not dup

        # Baska bir oturumdan BAKILDIGINDA henuz gorunmemeli (commit yok).
        other = SessionLocal()
        try:
            assert other.query(Photo).filter(Photo.id == photo_id).first() is None, (
                "save_upload kendi basina COMMIT ETMEMELI"
            )
        finally:
            other.close()

        db.rollback()
    finally:
        db.close()

    # Rollback sonrasi kalici olmamali
    check = SessionLocal()
    try:
        assert check.query(Photo).filter(Photo.id == photo_id).first() is None
    finally:
        check.close()


def test_job_insert_failure_rolls_back_photo(test_user_id):
    """Job insert patlarsa foto insert de GERI ALINMALI."""
    db = SessionLocal()
    photo_id = None
    try:
        photo, _ = photo_service.save_upload(db, _upload_file(f"b-{uuid.uuid4()}.jpg"),
                                             test_user_id)
        photo_id = photo.id

        jr.enqueue(db, JOB_TYPE_FACE_PIPELINE, {"photo_id": str(photo_id)}, test_user_id)

        # Ikinci job'i BILEREK patlat: var olmayan kullanici -> FK ihlali
        with pytest.raises(Exception):
            jr.enqueue(db, JOB_TYPE_VLM_ANALYSIS, {"photo_id": str(photo_id)},
                       uuid.uuid4())
            db.commit()

        db.rollback()
    finally:
        db.close()

    check = SessionLocal()
    try:
        assert check.query(Photo).filter(Photo.id == photo_id).first() is None, (
            "Job insert basarisiz olunca foto satiri da GERI ALINMALIYDI"
        )
        remaining = check.execute(
            text("SELECT count(*) FROM jobs WHERE payload->>'photo_id' = :p"),
            {"p": str(photo_id)},
        ).scalar_one()
        assert remaining == 0, "Basarili olan ilk job da geri alinmaliydi"
    finally:
        check.close()


def test_successful_upload_commits_photo_and_both_jobs(test_user_id):
    """Mutlu yol: foto + IKI job birlikte kalici olur."""
    db = SessionLocal()
    photo_id = None
    try:
        photo, _ = photo_service.save_upload(db, _upload_file(f"c-{uuid.uuid4()}.jpg"),
                                             test_user_id)
        photo_id = photo.id
        face_job = jr.enqueue(db, JOB_TYPE_FACE_PIPELINE, {"photo_id": str(photo_id)},
                              test_user_id)
        vlm_job = jr.enqueue(db, JOB_TYPE_VLM_ANALYSIS, {"photo_id": str(photo_id)},
                             test_user_id)
        db.commit()
    finally:
        db.close()

    try:
        check = SessionLocal()
        try:
            assert check.query(Photo).filter(Photo.id == photo_id).first() is not None
        finally:
            check.close()

        assert jr.get_status(face_job)["status"] == "queued"
        assert jr.get_status(vlm_job)["status"] == "queued"
        assert jr.get_status(face_job)["type"] == JOB_TYPE_FACE_PIPELINE
        assert jr.get_status(vlm_job)["type"] == JOB_TYPE_VLM_ANALYSIS
    finally:
        _cleanup(photo_id, test_user_id)
