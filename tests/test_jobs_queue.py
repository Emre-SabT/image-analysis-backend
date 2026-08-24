"""Is kuyrugu davranis testleri - GERCEK PostgreSQL'e karsi calisir.

SKIP LOCKED, sahiplik kosulu ve yaris durumlari mock'lanamaz: bunlarin
tamami veritabani motorunun davranisidir. Bu yuzden testler canli semaya
karsi kosar ve kendi olusturdugu satirlari temizler.
"""

import threading
import uuid

import pytest
from sqlalchemy import text

from app.db import jobs_repository as jr
from app.db.models import JOB_TYPE_FACE_PIPELINE, JOB_TYPE_VLM_ANALYSIS
from app.db.session import SessionLocal


def _enqueue(user_id, job_type=JOB_TYPE_VLM_ANALYSIS, priority=0):
    db = SessionLocal()
    try:
        job_id = jr.enqueue(db, job_type, {"photo_id": str(uuid.uuid4())},
                            user_id, priority)
        db.commit()
        return job_id
    finally:
        db.close()


def _set_locked_at(job_id, expr):
    db = SessionLocal()
    try:
        db.execute(text(f"UPDATE jobs SET locked_at = {expr} WHERE id = :i"),
                   {"i": str(job_id)})
        db.commit()
    finally:
        db.close()


def _row(job_id):
    return jr.get_status(job_id)


# --- SKIP LOCKED: iki worker ayni isi almaz ----------------------------


def test_two_workers_never_claim_same_job(test_user_id):
    _enqueue(test_user_id)
    _enqueue(test_user_id)

    claimed = []
    lock = threading.Lock()

    def claim(worker):
        job = jr.claim_next(worker, [JOB_TYPE_VLM_ANALYSIS])
        if job:
            with lock:
                claimed.append((worker, job.id))

    threads = [threading.Thread(target=claim, args=(f"w{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ids = [job_id for _, job_id in claimed]
    assert len(ids) == len(set(ids)), "Iki worker AYNI isi almis olamaz"


def test_claim_marks_running_and_increments_attempts(test_user_id):
    job_id = _enqueue(test_user_id)
    job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
    assert job is not None
    row = _row(job.id)
    assert row["status"] == "running"
    assert row["attempts"] == 1, "attempts CLAIM aninda artmali"
    assert row["started_at"] is not None


# --- Sahiplik kosulu: stale worker sonucu ezemez -----------------------


def test_stale_worker_cannot_complete_after_reap(test_user_id):
    _enqueue(test_user_id)
    job = jr.claim_next("eski-worker", [JOB_TYPE_VLM_ANALYSIS])
    assert job is not None

    # Isi bayatlat ve reaper'i calistir
    _set_locked_at(job.id, "now() - interval '10 minutes'")
    assert jr.reap_stale(300) >= 1
    assert _row(job.id)["status"] == "queued", "Reaper isi kuyruga geri koymali"

    # Eski worker artik sahibi degil
    assert jr.complete(job.id, "eski-worker") is False
    assert jr.fail(job.id, "eski-worker", "gec kalan hata") is False
    assert jr.heartbeat(job.id, "eski-worker") is False

    # Yeni worker devralip tamamlayabilir
    new_job = jr.claim_next("yeni-worker", [JOB_TYPE_VLM_ANALYSIS])
    assert new_job is not None and new_job.id == job.id
    assert jr.complete(new_job.id, "yeni-worker") is True
    assert _row(job.id)["status"] == "done"


def test_heartbeat_keeps_job_alive_then_reaper_takes_it(test_user_id):
    _enqueue(test_user_id)
    job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
    assert job is not None

    # Timeout'tan ESKI gorunsun ama heartbeat calissin -> reaper ALMAMALI
    _set_locked_at(job.id, "now() - interval '10 minutes'")
    assert jr.heartbeat(job.id, "w1") is True  # locked_at tazelendi
    assert jr.reap_stale(300) == 0, "Heartbeat atan is stale sayilMAMALI"
    assert _row(job.id)["status"] == "running"

    # Heartbeat dursun (bayatlat) -> reaper ALMALI
    _set_locked_at(job.id, "now() - interval '10 minutes'")
    assert jr.reap_stale(300) == 1
    assert _row(job.id)["status"] == "queued"


# --- Adil siralama ------------------------------------------------------


def test_fair_ordering_interleaves_users(db_session):
    """Kullanici A 10 is, B 2 is -> claim sirasi IC ICE gecmeli."""
    from app.db.models import User

    ids = {}
    for name in ("A", "B"):
        uid = uuid.uuid4()
        db_session.add(User(id=uid, email=f"fair-{name}-{uid}@t.internal",
                            password_hash="x", display_name=name, role="editor"))
        ids[name] = uid
    db_session.commit()

    try:
        job_owner = {}
        for _ in range(10):
            job_owner[_enqueue(ids["A"], JOB_TYPE_FACE_PIPELINE)] = "A"
        for _ in range(2):
            job_owner[_enqueue(ids["B"], JOB_TYPE_FACE_PIPELINE)] = "B"

        order = []
        while True:
            job = jr.claim_next("fair-w", [JOB_TYPE_FACE_PIPELINE])
            if job is None:
                break
            if job.id in job_owner:
                order.append(job_owner[job.id])
            jr.complete(job.id, "fair-w")

        # B'nin 2 isi, A'nin 10 isinin SONUNA kalmamali.
        b_positions = [i for i, o in enumerate(order) if o == "B"]
        assert b_positions, "B'nin isleri alinmali"
        assert max(b_positions) < 4, (
            f"B'nin isleri one karismali, sirasi: {order}"
        )
    finally:
        for uid in ids.values():
            db_session.execute(text("DELETE FROM jobs WHERE user_id=:u"), {"u": str(uid)})
            db_session.execute(text("DELETE FROM user_job_counters WHERE user_id=:u"), {"u": str(uid)})
            db_session.execute(text("DELETE FROM users WHERE id=:u"), {"u": str(uid)})
        db_session.commit()


# --- Eszamanli enqueue: sequence tekil ve ardisik -----------------------


def test_concurrent_enqueue_produces_unique_sequences(test_user_id):
    """20 paralel enqueue -> sequence_in_user TEKIL ve ARDISIK olmali.

    COUNT(*) kullanilsaydi bu test yaris durumu nedeniyle duserdi.
    """
    errors = []

    def worker():
        try:
            _enqueue(test_user_id, JOB_TYPE_FACE_PIPELINE)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"enqueue hatalari: {errors}"

    db = SessionLocal()
    try:
        seqs = [r[0] for r in db.execute(
            text("SELECT sequence_in_user FROM jobs WHERE user_id=:u ORDER BY sequence_in_user"),
            {"u": str(test_user_id)},
        ).fetchall()]
    finally:
        db.close()

    assert len(seqs) == 20
    assert len(set(seqs)) == 20, f"sequence_in_user TEKIL olmali: {seqs}"
    assert seqs == list(range(1, 21)), f"ARDISIK olmali: {seqs}"


# --- Retry / backoff ----------------------------------------------------


def test_retry_increments_attempts_and_pushes_run_after(test_user_id):
    _enqueue(test_user_id)
    job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
    assert jr.fail(job.id, "w1", "gecici hata", retry=True) is True

    row = _row(job.id)
    assert row["status"] == "queued", "Deneme hakki varken kuyruga donmeli"
    assert row["attempts"] == 1
    assert row["error"] == "gecici hata"
    assert row["run_after"] > row["created_at"], "run_after ileri atilmali (backoff)"


def test_exhausted_attempts_marks_failed(test_user_id):
    _enqueue(test_user_id)
    job_id = None
    for expected_attempt in (1, 2, 3):
        job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
        if job is None:
            # backoff nedeniyle run_after ileride - testi hizlandirmak icin sifirla
            _reset_run_after(job_id)
            job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
        assert job is not None
        job_id = job.id
        assert job.attempts == expected_attempt
        jr.fail(job.id, "w1", f"hata {expected_attempt}", retry=True)

    row = _row(job_id)
    assert row["status"] == "failed", "max_attempts asilinca 'failed' olmali"
    assert row["attempts"] == 3
    assert row["finished_at"] is not None


def _reset_run_after(job_id):
    db = SessionLocal()
    try:
        db.execute(text("UPDATE jobs SET run_after = now() - interval '1 minute' WHERE id=:i"),
                   {"i": str(job_id)})
        db.commit()
    finally:
        db.close()


def test_no_retry_marks_failed_immediately(test_user_id):
    _enqueue(test_user_id)
    job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
    assert jr.fail(job.id, "w1", "kalici hata", retry=False) is True
    assert _row(job.id)["status"] == "failed"


def test_backoff_delays_next_claim(test_user_id):
    _enqueue(test_user_id)
    job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
    jr.fail(job.id, "w1", "hata", retry=True)
    # run_after ileride oldugu icin hemen tekrar alinmamali
    assert jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS]) is None


# --- Iki job tipinin bagimsizligi ---------------------------------------


def test_vlm_failure_does_not_affect_face_job(test_user_id):
    """Ayni fotografin iki isi TAMAMEN BAGIMSIZ olmali."""
    photo_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        face_id = jr.enqueue(db, JOB_TYPE_FACE_PIPELINE, {"photo_id": photo_id}, test_user_id)
        vlm_id = jr.enqueue(db, JOB_TYPE_VLM_ANALYSIS, {"photo_id": photo_id}, test_user_id)
        db.commit()
    finally:
        db.close()

    vlm = jr.claim_next("w-vlm", [JOB_TYPE_VLM_ANALYSIS])
    assert vlm.id == vlm_id
    jr.fail(vlm.id, "w-vlm", "VLM coktu", retry=False)
    assert _row(vlm_id)["status"] == "failed"

    # face isi hicbir sekilde etkilenmemeli
    assert _row(face_id)["status"] == "queued"
    face = jr.claim_next("w-face", [JOB_TYPE_FACE_PIPELINE])
    assert face.id == face_id
    assert jr.complete(face.id, "w-face") is True
    assert _row(face_id)["status"] == "done"
    assert _row(vlm_id)["status"] == "failed", "face basarisi VLM'i degistirmemeli"


def test_worker_only_claims_its_own_type(test_user_id):
    _enqueue(test_user_id, JOB_TYPE_VLM_ANALYSIS)
    assert jr.claim_next("w-face", [JOB_TYPE_FACE_PIPELINE]) is None, (
        "worker-face bir VLM isini ASLA almamali"
    )


# --- Sorgular -----------------------------------------------------------


def test_count_queued_for_user(test_user_id):
    for _ in range(3):
        _enqueue(test_user_id)
    assert jr.count_queued_for_user(test_user_id) == 3


def test_queue_status_reports_per_type_not_mixed(test_user_id):
    """face ve vlm sureleri cok farkli - KARISIK TEK ORTALAMA olmamali."""
    _enqueue(test_user_id, JOB_TYPE_VLM_ANALYSIS)
    _enqueue(test_user_id, JOB_TYPE_FACE_PIPELINE)
    _enqueue(test_user_id, JOB_TYPE_FACE_PIPELINE)

    status = jr.queue_status_by_type(test_user_id, 20)
    assert status[JOB_TYPE_VLM_ANALYSIS]["queued"] == 1
    assert status[JOB_TYPE_FACE_PIPELINE]["queued"] == 2


def test_claim_next_rejects_empty_job_types():
    with pytest.raises(ValueError):
        jr.claim_next("w1", [])
