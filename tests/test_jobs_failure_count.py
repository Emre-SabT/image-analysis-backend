"""failure_count / attempts ayriminin regresyon testleri (PR-1).

Bkz. jobs_repository.py basindaki "ATTEMPTS / FAILURE_COUNT AYRIMI" notu ve
migration f1a2b3c4d5e6. Bu testler GERCEK PostgreSQL'e karsi calisir (diger
is kuyrugu testleriyle ayni desen, bkz. test_jobs_queue.py).
"""

import uuid

from app.db import jobs_repository as jr
from app.db.models import JOB_TYPE_VLM_ANALYSIS
from app.db.session import SessionLocal
from sqlalchemy import text


def _enqueue(user_id, job_type=JOB_TYPE_VLM_ANALYSIS, priority=0):
    db = SessionLocal()
    try:
        job_id = jr.enqueue(db, job_type, {"photo_id": str(uuid.uuid4())},
                            user_id, priority)
        db.commit()
        return job_id
    finally:
        db.close()


def _row(job_id):
    return jr.get_status(job_id)


def _reset_run_after(job_id):
    """Backoff nedeniyle ileride kalan run_after'i testte hizlandirir."""
    db = SessionLocal()
    try:
        db.execute(text("UPDATE jobs SET run_after = now() - interval '1 minute' WHERE id=:i"),
                   {"i": str(job_id)})
        db.commit()
    finally:
        db.close()


# --- Asil regresyon testi: lock-cakismasi retry butcesini TUKETMEMELI ---


def test_lock_conflict_requeue_does_not_consume_retry_budget(test_user_id):
    """Bug kaniti: attempts HER claim'de artar (reaper'in yanlis-pozitif
    reclaim'i / lock-cakismasi requeue'su DAHIL), ama bunlarin hicbiri
    GERCEK bir deneme degildir. Duzeltmeden ONCE (attempts uzerinden
    max_attempts kontrolu), bu senaryoda is 4. claim'de - ilk GERCEK
    hatasinda - haksiz yere kalici 'failed' olurdu.

    Senaryo: is 3 kez claim edilip HER SEFERINDE requeue_lock_conflict ile
    (attempts+1, failure_count SABIT) cezasiz kuyruga donuyor, sonra 4.
    claim'de GERCEK bir hata olusuyor (fail cagriliyor).
    """
    job_id = _enqueue(test_user_id)

    for _ in range(3):
        job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
        assert job is not None
        assert jr.requeue_lock_conflict(job.id, "w1", delay_seconds=0) is True
        _reset_run_after(job.id)

    row = _row(job_id)
    assert row["attempts"] == 3, "attempts her claim'de artmis olmali"
    assert row["failure_count"] == 0, "lock-cakismasi GERCEK hata SAYILMAMALI"
    assert row["status"] == "queued"

    job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
    assert job is not None
    assert job.attempts == 4, "attempts claim sayaci olmaya devam ediyor (max_attempts'i ASMIS durumda)"

    assert jr.fail(job.id, "w1", "ilk GERCEK hata", retry=True) is True

    row = _row(job_id)
    assert row["status"] == "queued", (
        "attempts(4) max_attempts'i (3) asmis olsa da bu job'in GERCEK "
        "ilk hatasi - failure_count(1) < max_attempts(3) oldugu icin "
        "'failed' OLMAMALI (PR-1'in duzelttigi bug tam olarak bu)"
    )
    assert row["failure_count"] == 1, "SADECE gercek fail() cagrisi failure_count'u artirmali"


# --- Failure_count'un kendi basina dogru calistigini kanitla ------------


def test_three_real_failures_marks_failed_with_failure_count(test_user_id):
    """3 GERCEK fail() cagrisi -> kalici 'failed', failure_count==3 (attempts
    de bu senaryoda 3'e esit olur ama karsilastirma artik failure_count
    uzerinden yapiliyor - bu test o kolonu ayrica dogrular)."""
    job_id = None
    _enqueue_job_id = _enqueue(test_user_id)
    job_id = _enqueue_job_id

    for expected in (1, 2, 3):
        job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
        if job is None:
            _reset_run_after(job_id)
            job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
        assert job is not None
        assert jr.fail(job.id, "w1", f"gercek hata {expected}", retry=True) is True

    row = _row(job_id)
    assert row["status"] == "failed"
    assert row["failure_count"] == 3
    assert row["attempts"] == 3


def test_requeue_lock_conflict_ignores_stale_ownership(test_user_id):
    """Sahiplik kosulu requeue_lock_conflict icin de gecerli olmali - eski
    worker artik sahibi degilse (baskasi devralmissa) False donmeli, hicbir
    satiri etkilememeli."""
    job_id = _enqueue(test_user_id)
    job = jr.claim_next("eski-worker", [JOB_TYPE_VLM_ANALYSIS])
    assert job is not None

    # Baska bir worker devralmis gibi simule et.
    db = SessionLocal()
    try:
        db.execute(text("UPDATE jobs SET locked_by = 'yeni-worker' WHERE id = :i"),
                   {"i": str(job_id)})
        db.commit()
    finally:
        db.close()

    assert jr.requeue_lock_conflict(job.id, "eski-worker", delay_seconds=0) is False
    row = _row(job_id)
    assert row["status"] == "running", "sahiplik kaybeden worker hicbir seyi degistirmemeli"
