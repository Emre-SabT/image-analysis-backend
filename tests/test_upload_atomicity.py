"""Ingestion'in atomikligi: foto + exif + 2 job TEK transaction'da.

Bu invaryant ONCEDEN photo_service.save_upload + POST /photos'ta yasiyordu
(save_upload commit DEGIL flush yapiyordu, commit router'daydi). Inbox ->
ingestion mimarisine gecisle AYNI invaryant ingestion_service._create_photo'ya
tasindi: job insert patlarsa foto satiri da kalici OLMAMALI, aksi halde
sistemde ASLA islenmeyecek "oksuz" fotograflar birikir.

AYRICA burada test edilen, eski testte OLMAYAN bir garanti var: commit
basarisiz olursa dosya stored/'da BIRAKILMAZ, inbox'a geri alinir - yoksa
hicbir kaydin isaret etmedigi oksuz bir dosya kalirdi.
"""

import io
import os
import uuid
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import text

from app.db.models import Photo
from app.db.session import SessionLocal
from app.services import ingestion_service, photo_service


def _make_inbox_file(user_id, name=None) -> tuple[Path, uuid.UUID]:
    """receive_upload'in urettigi ile AYNI sekli elle olusturur:
    inbox/{photo_id}.jpg + yan-dosya.

    Goruntu icerigi HER CAGRIDA RASTGELE: content_hash artik UNIQUE oldugu
    icin sabit bir test goruntusu (or. duz kirmizi 4x4) her kosuda AYNI
    hash'i uretir ve ikinci kosu 'duplicate' donerdi - test kendi kendini
    bozardi."""
    photo_id = uuid.uuid4()
    name = name or f"atomicity-{photo_id}.jpg"
    image_path = photo_service.INBOX_DIR / f"{photo_id}.jpg"

    img = Image.new("RGB", (8, 8))
    img.putdata([tuple(os.urandom(3)) for _ in range(64)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    image_path.write_bytes(data)

    photo_service._write_inbox_meta(
        image_path,
        {
            "photo_id": str(photo_id),
            "original_filename": name,
            "content_hash": ingestion_service.hash_file(image_path),
            "size_bytes": len(data),
            "uploaded_by_user_id": str(user_id),
            "received_at": "2026-09-04T00:00:00",
        },
    )
    return image_path, photo_id


def _cleanup(photo_id, user_id):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM jobs WHERE user_id=:u"), {"u": str(user_id)})
        db.execute(text("DELETE FROM activity_log WHERE target_id=:p"), {"p": str(photo_id)})
        db.execute(text("DELETE FROM photo_exif WHERE photo_id=:p"), {"p": str(photo_id)})
        db.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(photo_id)})
        db.commit()
    finally:
        db.close()
    for p in (
        photo_service.STORED_DIR / f"{photo_id}.jpg",
        photo_service.INBOX_DIR / f"{photo_id}.jpg",
    ):
        p.unlink(missing_ok=True)
        photo_service._meta_path_for(p).unlink(missing_ok=True)


def test_successful_ingest_commits_photo_and_both_jobs(test_user_id):
    """Mutlu yol: foto + IKI job birlikte kalici olur, dosya stored/'a gecer."""
    image_path, photo_id = _make_inbox_file(test_user_id)
    try:
        result = ingestion_service.ingest_file(image_path)
        assert result.outcome == ingestion_service.IngestOutcome.CREATED, f"detay={result.detail!r}"
        assert result.photo_id == photo_id

        check = SessionLocal()
        try:
            photo = check.query(Photo).filter(Photo.id == photo_id).first()
            assert photo is not None, "Photo satiri commit edilmeliydi"
            assert photo.storage_path.endswith(f"{photo_id}.jpg")

            jobs = check.execute(
                text("SELECT type FROM jobs WHERE payload->>'photo_id' = :p ORDER BY type"),
                {"p": str(photo_id)},
            ).scalars().all()
            assert jobs == ["face_pipeline", "vlm_analysis"], f"beklenmeyen is kaydi: {jobs}"
        finally:
            check.close()

        assert (photo_service.STORED_DIR / f"{photo_id}.jpg").exists(), "dosya stored/'a tasinmaliydi"
        assert not image_path.exists(), "inbox'ta kalmamaliydi"
        assert not photo_service._meta_path_for(image_path).exists(), (
            "meta yan-dosyasi ingestion bitince SILINMELI - varligi 'bitmedi' isaretidir"
        )
    finally:
        _cleanup(photo_id, test_user_id)


def test_job_insert_failure_rolls_back_photo_and_restores_file(test_user_id, monkeypatch):
    """Job insert patlarsa: foto satiri kalici OLMAMALI ve dosya inbox'a DONMELI."""
    image_path, photo_id = _make_inbox_file(test_user_id)
    try:
        real_enqueue = ingestion_service.jobs_repository.enqueue
        calls = {"n": 0}

        def flaky_enqueue(session, job_type, payload, user_id, priority=0):
            calls["n"] += 1
            if calls["n"] == 2:  # ilk is gecer, IKINCI is patlar
                raise RuntimeError("bilerek patlatildi (vlm_analysis enqueue)")
            return real_enqueue(session, job_type, payload, user_id, priority)

        monkeypatch.setattr(ingestion_service.jobs_repository, "enqueue", flaky_enqueue)

        result = ingestion_service.ingest_file(image_path)
        assert result.outcome == ingestion_service.IngestOutcome.FAILED

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

        assert image_path.exists(), (
            "commit basarisizsa dosya inbox'a GERI ALINMALI - stored/'da oksuz "
            "dosya birakmak, hicbir kaydin isaret etmedigi veri demektir"
        )
        assert not (photo_service.STORED_DIR / f"{photo_id}.jpg").exists()
    finally:
        _cleanup(photo_id, test_user_id)


def test_ingest_is_retriable_after_transient_failure(test_user_id, monkeypatch):
    """TEST 5 - transaction sirasinda crash: dosya inbox'ta kalir ve
    SONRAKI turda bastan islenip basariyla tamamlanir."""
    image_path, photo_id = _make_inbox_file(test_user_id)
    try:
        real_enqueue = ingestion_service.jobs_repository.enqueue

        def boom(*a, **kw):
            raise RuntimeError("DB gitti (simule)")

        monkeypatch.setattr(ingestion_service.jobs_repository, "enqueue", boom)
        assert ingestion_service.ingest_file(image_path).outcome == \
            ingestion_service.IngestOutcome.FAILED
        assert image_path.exists(), "gecici hatada dosya inbox'ta KALMALI"

        # "Servis yeniden basladi": gercek enqueue geri geldi.
        monkeypatch.setattr(ingestion_service.jobs_repository, "enqueue", real_enqueue)
        result = ingestion_service.ingest_file(image_path)
        assert result.outcome == ingestion_service.IngestOutcome.CREATED, f"detay={result.detail!r}"

        check = SessionLocal()
        try:
            assert check.query(Photo).filter(Photo.id == photo_id).first() is not None
        finally:
            check.close()
    finally:
        _cleanup(photo_id, test_user_id)


def test_crash_after_commit_before_move_is_recovered(test_user_id):
    """TEST 4/senaryo D - DB commit oldu ama dosya tasinamadan crash.

    Restart'ta ayni dosya yeniden gorulur; sistem bunu 'kullanici ayni
    fotografi iki kez yukledi' SANMAMALI - yarim kalmis tasimayi
    tamamlamali, YENI kayit/is OLUSTURMAMALIDIR.
    """
    image_path, photo_id = _make_inbox_file(test_user_id)
    try:
        assert ingestion_service.ingest_file(image_path).outcome == \
            ingestion_service.IngestOutcome.CREATED

        stored = photo_service.STORED_DIR / f"{photo_id}.jpg"
        db = SessionLocal()
        try:
            jobs_before = db.execute(
                text("SELECT count(*) FROM jobs WHERE payload->>'photo_id' = :p"),
                {"p": str(photo_id)},
            ).scalar_one()
        finally:
            db.close()

        # CRASH SIMULASYONU: kayit commit'li ama dosya nihai yerinde degil -
        # dosyayi inbox'a geri koyuyoruz (tasimanin yarim kalmis hali).
        os.replace(stored, image_path)
        photo_service._write_inbox_meta(
            image_path,
            {
                "photo_id": str(photo_id),
                "original_filename": "orange.jpg",
                "content_hash": ingestion_service.hash_file(image_path),
                "size_bytes": image_path.stat().st_size,
                "uploaded_by_user_id": str(test_user_id),
                "received_at": "2026-09-04T00:00:00",
            },
        )

        result = ingestion_service.ingest_file(image_path)
        assert result.outcome == ingestion_service.IngestOutcome.RECOVERED, (
            f"yarim tasima 'recovered' olmaliydi, {result.outcome} dondu"
        )
        assert stored.exists(), "dosya nihai yerine tasinmaliydi"

        db = SessionLocal()
        try:
            assert db.execute(
                text("SELECT count(*) FROM photos WHERE content_hash = "
                     "(SELECT content_hash FROM photos WHERE id = :p)"),
                {"p": str(photo_id)},
            ).scalar_one() == 1, "IKINCI bir Photo satiri olusmamaliydi"
            jobs_after = db.execute(
                text("SELECT count(*) FROM jobs WHERE payload->>'photo_id' = :p"),
                {"p": str(photo_id)},
            ).scalar_one()
            assert jobs_after == jobs_before, "YENI is kaydi olusmamaliydi"
        finally:
            db.close()
    finally:
        _cleanup(photo_id, test_user_id)
