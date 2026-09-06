"""ITEM 3 - DB ve depolama arizalarindan kurtarma.

Her test GERCEK bir ariza bicimini taklit eder ve tek bir soruyu yanitlar:
ariza gectikten sonra sistem KENDILIGINDEN duzeliyor mu, veri kayboluyor mu?

Ariza taklidi neden monkeypatch ile: gercekten diski doldurmak ya da
Postgres'i durdurmak bu makinedeki CALISAN sistemi etkilerdi. Taklit,
arizanin kodda gorundugu YERE (os.replace / commit / enqueue) yerlestiriliyor -
yani kurtarma yolu gercekten kosuluyor, atlanmiyor.
"""

import io
import os
import uuid
from pathlib import Path

from PIL import Image
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session as SASession

from app.db.models import Photo
from app.db.session import SessionLocal
from app.services import ingestion_service, photo_service
from app.services.ingestion_service import IngestOutcome


def _jpeg() -> bytes:
    img = Image.new("RGB", (8, 8))
    img.putdata([tuple(os.urandom(3)) for _ in range(64)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _place(user_id, data=None) -> tuple[Path, uuid.UUID]:
    pid = uuid.uuid4()
    data = data if data is not None else _jpeg()
    p = photo_service.INBOX_DIR / f"{pid}.jpg"
    p.write_bytes(data)
    photo_service._write_inbox_meta(p, {
        "photo_id": str(pid),
        "original_filename": f"failrec-{pid}.jpg",
        "content_hash": ingestion_service.hash_file(p),
        "size_bytes": len(data),
        "uploaded_by_user_id": str(user_id),
        "received_at": "2026-09-04T00:00:00",
    })
    return p, pid


def _purge(ids, user_id):
    db = SessionLocal()
    try:
        for pid in ids:
            db.execute(text("DELETE FROM jobs WHERE payload->>'photo_id' = :p"), {"p": str(pid)})
            db.execute(text("DELETE FROM activity_log WHERE target_id=:p"), {"p": str(pid)})
            db.execute(text("DELETE FROM photo_exif WHERE photo_id=:p"), {"p": str(pid)})
            db.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(pid)})
        db.execute(text("DELETE FROM jobs WHERE user_id = :u"), {"u": str(user_id)})
        db.commit()
    finally:
        db.close()
    for pid in ids:
        for d in (photo_service.STORED_DIR, photo_service.INBOX_DIR):
            (d / f"{pid}.jpg").unlink(missing_ok=True)
            photo_service._meta_path_for(d / f"{pid}.jpg").unlink(missing_ok=True)


# --- DB arizasi --------------------------------------------------------


def test_db_down_during_ingest_leaves_file_and_recovers(test_user_id, monkeypatch):
    """DB ERISILEMEZ oldugunda dosya KAYBOLMAZ ve DB donunce islenir.

    Kritik nokta: dosya failed/'a ATILMAMALI. Gecici bir altyapi arizasi
    yuzunden fotografi kalici olarak "islenemez" kutusuna koymak, operator
    mudahalesi olmadan asla geri donmeyecek bir veri hapsi olurdu.
    """
    image_path, photo_id = _place(test_user_id)
    try:
        def db_down(*a, **kw):
            raise OperationalError("INSERT", {}, Exception("server closed the connection"))

        monkeypatch.setattr(ingestion_service.jobs_repository, "enqueue", db_down)
        r = ingestion_service.ingest_file(image_path)
        assert r.outcome == IngestOutcome.FAILED

        assert image_path.exists(), "DB arizasinda dosya INBOX'TA KALMALI"
        assert not (photo_service.FAILED_DIR / image_path.name).exists(), (
            "gecici DB arizasi dosyayi failed/'a ATMAMALI - kalici veri hapsi olurdu"
        )
        db = SessionLocal()
        try:
            assert db.query(Photo).filter(Photo.id == photo_id).first() is None
        finally:
            db.close()

        monkeypatch.undo()  # DB geri geldi
        r2 = ingestion_service.ingest_file(image_path)
        assert r2.outcome == IngestOutcome.CREATED, f"DB donunce islenmeliydi: {r2.detail}"
    finally:
        _purge([photo_id], test_user_id)


def test_commit_failure_leaves_no_orphan_file_or_row(test_user_id, monkeypatch):
    """Commit patlarsa NE oksuz satir NE oksuz dosya kalmali."""
    image_path, photo_id = _place(test_user_id)
    stored = photo_service.STORED_DIR / f"{photo_id}.jpg"
    try:
        def boom(self):
            raise OperationalError("COMMIT", {}, Exception("could not write: No space left"))

        monkeypatch.setattr(SASession, "commit", boom)
        r = ingestion_service.ingest_file(image_path)
        monkeypatch.undo()

        assert r.outcome == IngestOutcome.FAILED
        assert not stored.exists(), "commit basarisizsa stored/'da OKSUZ DOSYA kalmamali"
        assert image_path.exists(), "dosya inbox'a geri alinmali"

        db = SessionLocal()
        try:
            assert db.query(Photo).filter(Photo.id == photo_id).first() is None
        finally:
            db.close()

        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
    finally:
        _purge([photo_id], test_user_id)


# --- Depolama arizasi --------------------------------------------------


def test_storage_move_failure_leaves_file_in_inbox(test_user_id, monkeypatch):
    """stored/'a tasima patlarsa (disk dolu / izin / AV kilidi) dosya
    inbox'ta kalmali ve DB'ye YARIM kayit girmemeli."""
    image_path, photo_id = _place(test_user_id)
    try:
        real_replace = os.replace

        def failing_replace(src, dst, *a, **kw):
            if str(photo_service.STORED_DIR) in str(dst):
                raise OSError(28, "No space left on device")
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(ingestion_service.os, "replace", failing_replace)
        r = ingestion_service.ingest_file(image_path)
        monkeypatch.undo()

        assert r.outcome == IngestOutcome.FAILED
        assert image_path.exists(), "tasima patlayinca dosya inbox'ta KALMALI"
        db = SessionLocal()
        try:
            assert db.query(Photo).filter(Photo.id == photo_id).first() is None, (
                "dosya tasinamamisken DB kaydi OLUSMAMALI"
            )
        finally:
            db.close()

        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
    finally:
        _purge([photo_id], test_user_id)


def test_unreadable_file_is_skipped_not_failed(test_user_id, monkeypatch):
    """Dosya O AN okunamiyorsa (AV taramasi, gecici kilit) SKIPPED olmali -
    sonraki turda tekrar denenir. FAILED olsaydi metrik yaniltirdi."""
    image_path, photo_id = _place(test_user_id)
    try:
        def unreadable(path):
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(ingestion_service, "hash_file", unreadable)
        r = ingestion_service.ingest_file(image_path)
        monkeypatch.undo()

        assert r.outcome == IngestOutcome.SKIPPED, f"{r.outcome} dondu"
        assert image_path.exists()
        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
    finally:
        _purge([photo_id], test_user_id)


def test_reconciliation_survives_storage_error(test_user_id, monkeypatch):
    """Uzlastirma sirasinda bir dosyada I/O hatasi olsa bile DIGERLERI
    islenmeli - tek arizali dosya tum crash kurtarmayi durdurmamali."""
    bad_img, bad_id = _place(test_user_id)
    good_img, good_id = _place(test_user_id)
    os.replace(bad_img, photo_service.STORED_DIR / bad_img.name)
    os.replace(good_img, photo_service.STORED_DIR / good_img.name)
    try:
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(5, "I/O error")
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(ingestion_service.os, "replace", flaky)
        ingestion_service.recover_orphan_meta()
        monkeypatch.undo()

        recovered = [p for p in (bad_img, good_img) if p.exists()]
        assert len(recovered) >= 1, (
            "tek dosyadaki I/O hatasi TUM uzlastirmayi durdurmamali - "
            "en az digeri kurtarilmaliydi"
        )
    finally:
        for pid in (bad_id, good_id):
            (photo_service.STORED_DIR / f"{pid}.jpg").unlink(missing_ok=True)
        _purge([bad_id, good_id], test_user_id)
