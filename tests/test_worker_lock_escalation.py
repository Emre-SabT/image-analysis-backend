"""LockConflict yolunda escalasyon esigi testleri (worker/main.py, PR-2).

job.attempts (SADECE claim sayisi - failure_count DEGIL, bkz. jobs_repository.py
basindaki not) settings.JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS'i asarsa
logger.error ile eskalasyon beklenir; altindaysa sessizce (info seviyesinde)
requeue edilir. Otomatik fail() TETIKLENMEMELI (bkz. settings.py gerekcesi).
"""

import logging
import uuid

from app.core.settings import settings
from app.db import jobs_repository as jr
from app.db.models import JOB_TYPE_FACE_PIPELINE
from app.db.session import SessionLocal
from app.services.photo_service import LockConflict
from app.worker import main as worker_main


def _enqueue(user_id):
    db = SessionLocal()
    try:
        job_id = jr.enqueue(db, JOB_TYPE_FACE_PIPELINE, {"photo_id": str(uuid.uuid4())}, user_id)
        db.commit()
        return job_id
    finally:
        db.close()


def _claim_and_bump_attempts(job_id, worker_id, attempts):
    """Gercekci ama testi hizlandiran kurulum: gercekten claim eder (locked_by
    dogru olsun diye - requeue_lock_conflict'in sahiplik kosulu icin sart),
    sonra attempts'i dogrudan istenen degere yukseltir (N kere claim
    dongusu kurmak yerine - PR-1 testlerindeki gibi gercek bir dongu kurmak
    burada GEREKSIZ: attempts'in KENDI mantigi zaten test_jobs_failure_count.py
    ve test_jobs_queue.py'de ayrica dogrulaniyor, burada sadece ESKALASYON
    KARARININ o deger uzerinden dogru calistigini test ediyoruz)."""
    from sqlalchemy import text

    job = jr.claim_next(worker_id, [JOB_TYPE_FACE_PIPELINE])
    assert job is not None
    db = SessionLocal()
    try:
        db.execute(text("UPDATE jobs SET attempts = :a WHERE id = :i"),
                   {"a": attempts, "i": str(job_id)})
        db.commit()
    finally:
        db.close()
    return jr.ClaimedJob(id=job.id, type=job.type, payload=job.payload,
                          user_id=job.user_id, attempts=attempts, max_attempts=job.max_attempts)


def test_below_threshold_does_not_escalate(test_user_id, caplog, monkeypatch):
    job_id = _enqueue(test_user_id)
    worker_id = "w-esik-alti"
    claimed = _claim_and_bump_attempts(job_id, worker_id,
                                        settings.JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS - 1)

    def raises_conflict(photo_id):
        raise LockConflict("test - kaynak mesgul")

    monkeypatch.setitem(worker_main.HANDLERS, JOB_TYPE_FACE_PIPELINE, raises_conflict)
    w = worker_main.Worker([JOB_TYPE_FACE_PIPELINE], worker_id=worker_id)

    with caplog.at_level(logging.ERROR):
        w._process(claimed)

    assert not any("KILIT CAKISMASI ESIGI ASILDI" in r.message for r in caplog.records)
    assert jr.get_status(job_id)["status"] == "queued", "cezasiz requeue calismali"
    assert jr.get_status(job_id)["failure_count"] == 0, "lock-conflict GERCEK hata sayilmamali"


def test_at_threshold_escalates_but_does_not_fail(test_user_id, caplog, monkeypatch):
    job_id = _enqueue(test_user_id)
    worker_id = "w-esik-ustu"
    claimed = _claim_and_bump_attempts(job_id, worker_id,
                                        settings.JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS)

    def raises_conflict(photo_id):
        raise LockConflict("test - kaynak mesgul")

    monkeypatch.setitem(worker_main.HANDLERS, JOB_TYPE_FACE_PIPELINE, raises_conflict)
    w = worker_main.Worker([JOB_TYPE_FACE_PIPELINE], worker_id=worker_id)

    with caplog.at_level(logging.ERROR):
        w._process(claimed)

    assert any("KILIT CAKISMASI ESIGI ASILDI" in r.message for r in caplog.records), (
        "esik asilinca logger.error ile ESKALE edilmeli"
    )
    row = jr.get_status(job_id)
    assert row["status"] == "queued", (
        "esik asilsa bile OTOMATIK fail EDILMEMELI - sadece eskale edilir "
        "(bkz. settings.py: sizinti tipik olarak worker restart ile duzelir)"
    )
    assert row["failure_count"] == 0
