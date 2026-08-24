"""PR-2'nin asil entegrasyon testi: ayni photo_id icin GERCEKTEN paralel
iki cagriyi calistirip advisory lock'un ikisinden SADECE birinin
detect_and_embed'e ulasmasini sagladigini kanitlar.

detect_and_embed MOCK'LANIYOR (gercek YuNet/AuraFace CALISTIRILMAZ) - testin
amaci kilit davranisi, model dogrulugu degil; gercek modelle bu test hem
yavas hem CI'da kirilgan olurdu.
"""

import threading
import time
import uuid

import pytest
from sqlalchemy import text

from app.db.models import Face, Photo
from app.db.session import SessionLocal
from app.services import face_service, photo_service
from app.services.photo_service import LockConflict


def _fake_detect_and_embed(db, photo):
    """Gercek face_service.detect_and_embed'in yerine gecer: TEK bir Face
    satiri uretip commit eder. Kilidin tuttugu pencereyi genisletmek icin
    (iki thread'in GERCEKTEN cakismasini garanti etmek amaciyla) kisa bir
    gecikme icerir - aksi halde mock o kadar hizli doner ki kaybeden thread
    kazanan zaten kilidi birakmisken acquire deneyebilir, bu da testi
    kacinilmaz olarak flaky yapardi."""
    time.sleep(0.5)
    face = Face(
        photo_id=photo.id,
        bbox={"x": 0, "y": 0, "w": 10, "h": 10},
        landmarks=[[0, 0]] * 5,
        det_confidence=0.9,
        crop_path="test-crop",
    )
    db.add(face)
    db.commit()
    return [face]


@pytest.fixture
def photo_row(test_user_id):
    db = SessionLocal()
    try:
        photo = Photo(filename="t.jpg", storage_path="uploads/t.jpg",
                      uploaded_by_user_id=test_user_id)
        db.add(photo)
        db.commit()
        db.refresh(photo)
        pid = photo.id
    finally:
        db.close()

    yield pid

    db2 = SessionLocal()
    try:
        db2.execute(text("DELETE FROM faces WHERE photo_id=:p"), {"p": str(pid)})
        db2.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(pid)})
        db2.commit()
    finally:
        db2.close()


def test_two_concurrent_calls_only_one_processes_the_photo(monkeypatch, photo_row):
    """Tam olarak bir thread Face uretir; digeri LockConflict alir (no-op
    DEGIL - worker/main.py'de bu, requeue_lock_conflict'e cevrilir; burada
    handler dogrudan cagrildigi icin exception olarak gozlemlenir)."""
    monkeypatch.setattr(face_service, "detect_and_embed", _fake_detect_and_embed)

    start_gate = threading.Barrier(2, timeout=5)
    results = {}

    def worker(name):
        try:
            start_gate.wait()  # iki thread'in GERCEKTEN ayni anda baslamasini garanti et
            photo_service.run_face_pipeline_job(photo_row)
            results[name] = "ok"
        except LockConflict:
            results[name] = "conflict"
        except Exception as exc:  # pragma: no cover
            results[name] = f"error: {exc}"

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("t1", "t2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert set(results.values()) == {"ok", "conflict"}, (
        f"tam olarak biri islemeli, digeri LockConflict almali: {results}"
    )

    db = SessionLocal()
    try:
        count = db.query(Face).filter(Face.photo_id == photo_row).count()
    finally:
        db.close()
    assert count == 1, "Face satiri TAM OLARAK BIR kez uretilmis olmali (duplicate YOK)"


def test_sequential_calls_second_is_idempotent_no_op(monkeypatch, photo_row):
    """Cakisma OLMADAN (kilit serbest kaldiktan SONRA) ayni is tekrar
    calisirsa - at-least-once'un GERCEK crash-recovery senaryosu - eski
    'already Face var mi' kontrolu hala gecerli: ikinci cagri sessizce
    no-op doner, LockConflict FIRLATMAZ (kilit bos, alinabiliyor)."""
    monkeypatch.setattr(face_service, "detect_and_embed", _fake_detect_and_embed)

    photo_service.run_face_pipeline_job(photo_row)  # ilk (gercek) calisma
    photo_service.run_face_pipeline_job(photo_row)  # ikinci (at-least-once tekrari)

    db = SessionLocal()
    try:
        count = db.query(Face).filter(Face.photo_id == photo_row).count()
    finally:
        db.close()
    assert count == 1, "sirali tekrar hala idempotent olmali (already-check)"
