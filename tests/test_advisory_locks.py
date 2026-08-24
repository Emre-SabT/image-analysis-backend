"""Photo-scoped advisory lock testleri (PR-2, app/db/locks.py).

Sayac riski, tek cagri noktasi (SADECE pg_try_advisory_lock) sayesinde
YAPISAL olarak ortadan kalktigi icin ayri bir "sayac" testi yok - onun
yerine bu invariant'in KENDISI (kod tabaninda pg_advisory_lock cagrisi
YOK) statik olarak dogrulanir (asagida).
"""

import logging
import uuid
from pathlib import Path

from app.db import locks
from app.db.session import SessionLocal


# --- Temel karsilikli-dislama davranisi ---------------------------------


def test_try_lock_blocks_same_classid_and_photo_id_across_sessions():
    db_a = SessionLocal()
    db_b = SessionLocal()
    photo_id = uuid.uuid4()
    try:
        assert locks.acquire_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id) is True
        assert locks.acquire_photo_lock(db_b, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id) is False
    finally:
        locks.release_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id)
        db_a.close()
        db_b.close()


def test_try_lock_different_photo_id_does_not_conflict():
    db_a = SessionLocal()
    db_b = SessionLocal()
    try:
        p1, p2 = uuid.uuid4(), uuid.uuid4()
        assert locks.acquire_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, p1) is True
        assert locks.acquire_photo_lock(db_b, locks.PHOTOAI_LOCK_CLASS_FACE, p2) is True
        locks.release_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, p1)
        locks.release_photo_lock(db_b, locks.PHOTOAI_LOCK_CLASS_FACE, p2)
    finally:
        db_a.close()
        db_b.close()


def test_try_lock_different_classid_does_not_conflict():
    """face_pipeline ve vlm_analysis AYNI photo_id icin birbirini BEKLEMEMELI."""
    db_a = SessionLocal()
    db_b = SessionLocal()
    photo_id = uuid.uuid4()
    try:
        assert locks.acquire_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id) is True
        assert locks.acquire_photo_lock(db_b, locks.PHOTOAI_LOCK_CLASS_VLM, photo_id) is True
        locks.release_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id)
        locks.release_photo_lock(db_b, locks.PHOTOAI_LOCK_CLASS_VLM, photo_id)
    finally:
        db_a.close()
        db_b.close()


def test_release_then_reacquire_by_another_session_succeeds():
    db_a = SessionLocal()
    db_b = SessionLocal()
    photo_id = uuid.uuid4()
    try:
        assert locks.acquire_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id) is True
        locks.release_photo_lock(db_a, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id)
        assert locks.acquire_photo_lock(db_b, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id) is True
        locks.release_photo_lock(db_b, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id)
    finally:
        db_a.close()
        db_b.close()


# --- release() gozlemlenebilirligi ---------------------------------------


def test_release_without_acquire_logs_warning(caplog):
    """Bende olmayan bir kilidi birakmaya calismak (lock_acquired bayraginin
    yanlis yonetildigi bir mantik hatasini temsil eder) sessizce GECILMEMELI."""
    db = SessionLocal()
    try:
        with caplog.at_level(logging.WARNING, logger="photoai.locks"):
            locks.release_photo_lock(db, locks.PHOTOAI_LOCK_CLASS_FACE, uuid.uuid4())
        assert any(
            "pg_advisory_unlock False dondu" in r.message for r in caplog.records
        ), "sahipsiz unlock denemesi warning olarak loglanmali"
    finally:
        db.close()


def test_release_exception_invalidates_session_and_logs_error(caplog, monkeypatch):
    """release() sirasinda baglanti hatasi olursa: exception YUTULUR (handler
    zaten isini bitirmis, release hatasi job'i fail etmemeli) AMA
    Session.invalidate() cagrilir VE logger.error yazilir - bu, sizmis
    kilidin en guclu tanisal sinyali (bkz. locks.py tasarim notu)."""
    db = SessionLocal()
    invalidated = {"called": False}

    def boom(*a, **kw):
        raise RuntimeError("baglanti koptu (simule)")

    def fake_invalidate():
        invalidated["called"] = True

    monkeypatch.setattr(db, "execute", boom)
    monkeypatch.setattr(db, "invalidate", fake_invalidate)

    with caplog.at_level(logging.ERROR, logger="photoai.locks"):
        locks.release_photo_lock(db, locks.PHOTOAI_LOCK_CLASS_FACE, uuid.uuid4())  # exception firlatMAMALI

    assert invalidated["called"], "release exception verince Session.invalidate() cagrilmali"
    assert any("SIZMIS KILIT" in r.message for r in caplog.records)


# --- Yapisal invariant: bloke olan cagri kod tabaninda HICBIR yerde yok --


def test_no_blocking_advisory_lock_calls_anywhere_in_app():
    """pg_advisory_lock (BLOKE OLAN, SAYACLI cagri) app/ agacinin HICBIR
    yerinde dogrudan kullanilmamali - SADECE pg_try_advisory_lock (tek
    cagri noktasi, locks.py). Bu, session-level advisory lock'un sayac
    riskini (ayni session ayni kilidi iki kez alip tek unlock'un
    sifirlamamasi) YAPISAL olarak ortadan kaldiran garantidir.

    Not: "pg_advisory_lock(" alt-dizisi ne "pg_try_advisory_lock(" ne de
    "pg_advisory_unlock(" icinde gecer (aralarina "try_"/"un" giriyor) -
    yani bu basit alt-dizi aramasi otomatik olarak dogru cagrilari
    (try/unlock) HARIC TUTAR, sadece dogrudan bloke-olan cagriyi yakalar.
    """
    app_root = Path(__file__).resolve().parent.parent / "app"
    offenders = []
    for path in app_root.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        if "pg_advisory_lock(" in content:
            offenders.append(str(path.relative_to(app_root.parent)))
    assert not offenders, (
        "Bloke olan pg_advisory_lock(...) dogrudan cagrilmis - SADECE "
        f"pg_try_advisory_lock kullanilmali. Bulunan dosyalar: {offenders}"
    )
