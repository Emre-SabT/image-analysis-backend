"""GET /photos/{id}/status ve /photos/status'un dondurdugu degerler.

Bes durumun HER BIRI ayri ayri dogrulanir: absent / queued / running /
done / failed. Ayrica yuz ve VLM'in BAGIMSIZLIGI - birinin 'failed'
olmasi digerinin degerini degistirmemeli.
"""

import uuid

from sqlalchemy import text

from app.db import jobs_repository as jr
from app.db.models import JOB_TYPE_FACE_PIPELINE, JOB_TYPE_VLM_ANALYSIS
from app.db.session import SessionLocal

ABSENT = jr.JOB_STATUS_ABSENT


def _enqueue_for_photo(photo_id, user_id, job_type):
    db = SessionLocal()
    try:
        job_id = jr.enqueue(db, job_type, {"photo_id": str(photo_id)}, user_id)
        db.commit()
        return job_id
    finally:
        db.close()


def _status(photo_id):
    return jr.photo_job_statuses([str(photo_id)])[str(photo_id)]


def _force_status(job_id, status):
    db = SessionLocal()
    try:
        db.execute(text("UPDATE jobs SET status = :s WHERE id = :i"),
                   {"s": status, "i": str(job_id)})
        db.commit()
    finally:
        db.close()


# --- Bes durumun her biri ----------------------------------------------


def test_absent_when_no_job_exists():
    """Kuyruk oncesi eski fotograf VEYA duplicate yukleme -> 'absent'.

    Bu ikisi ayni davranir: ikisinde de islenecek is YOK. (Duplicate
    bilgisi kullaniciya zaten POST /photos yanitindaki `duplicate: true`
    ile ayrica bildiriliyor.)
    """
    s = _status(uuid.uuid4())
    assert s["face_status"] == ABSENT
    assert s["vlm_status"] == ABSENT


def test_queued(test_user_id):
    photo_id = uuid.uuid4()
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_FACE_PIPELINE)
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_VLM_ANALYSIS)

    s = _status(photo_id)
    assert s["face_status"] == "queued"
    assert s["vlm_status"] == "queued"


def test_running(test_user_id):
    photo_id = uuid.uuid4()
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_FACE_PIPELINE)
    jr.claim_next("w1", [JOB_TYPE_FACE_PIPELINE])

    assert _status(photo_id)["face_status"] == "running"


def test_done(test_user_id):
    photo_id = uuid.uuid4()
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_FACE_PIPELINE)
    job = jr.claim_next("w1", [JOB_TYPE_FACE_PIPELINE])
    assert jr.complete(job.id, "w1") is True

    assert _status(photo_id)["face_status"] == "done"


def test_failed(test_user_id):
    photo_id = uuid.uuid4()
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_VLM_ANALYSIS)
    job = jr.claim_next("w1", [JOB_TYPE_VLM_ANALYSIS])
    assert jr.fail(job.id, "w1", "kalici hata", retry=False) is True

    assert _status(photo_id)["vlm_status"] == "failed"


# --- Bagimsizlik --------------------------------------------------------


def test_vlm_failure_leaves_face_status_untouched(test_user_id):
    """Kisit 3: birinin hatasi digerinin gosterimini ETKILEMEMELI."""
    photo_id = uuid.uuid4()
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_FACE_PIPELINE)
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_VLM_ANALYSIS)

    vlm = jr.claim_next("w-vlm", [JOB_TYPE_VLM_ANALYSIS])
    jr.fail(vlm.id, "w-vlm", "VLM coktu", retry=False)

    s = _status(photo_id)
    assert s["vlm_status"] == "failed"
    assert s["face_status"] == "queued", "VLM hatasi yuz durumunu degistirmemeli"

    face = jr.claim_next("w-face", [JOB_TYPE_FACE_PIPELINE])
    jr.complete(face.id, "w-face")

    s = _status(photo_id)
    assert s["face_status"] == "done"
    assert s["vlm_status"] == "failed", "Yuz basarisi VLM hatasini gizlememeli"


def test_mixed_states_on_same_photo(test_user_id):
    """Yuz bitmis, VLM hala sirada - kademeli gosterimin temel senaryosu."""
    photo_id = uuid.uuid4()
    face_job = _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_FACE_PIPELINE)
    _enqueue_for_photo(photo_id, test_user_id, JOB_TYPE_VLM_ANALYSIS)
    _force_status(face_job, "done")

    s = _status(photo_id)
    assert s["face_status"] == "done"
    assert s["vlm_status"] == "queued"


# --- Toplu sorgu --------------------------------------------------------


def test_batch_returns_entry_for_every_requested_id(test_user_id):
    """Istenen HER id icin bir kayit donmeli - hic isi olmayanlar dahil."""
    with_jobs = uuid.uuid4()
    _enqueue_for_photo(with_jobs, test_user_id, JOB_TYPE_FACE_PIPELINE)
    without_jobs = uuid.uuid4()

    result = jr.photo_job_statuses([str(with_jobs), str(without_jobs)])

    assert len(result) == 2
    assert result[str(with_jobs)]["face_status"] == "queued"
    assert result[str(without_jobs)]["face_status"] == ABSENT


def test_batch_handles_large_id_list(test_user_id):
    """Sunucu tarafi 50'den fazla id'yi tek sorguda karsilayabilmeli.

    (Parcalama ISTEMCI tarafinda yapiliyor - bkz. frontend chunkIds testi -
    ama sunucunun kendisi bir parca sinirina bagli olmamali.)
    """
    ids = [str(uuid.uuid4()) for _ in range(120)]
    tracked = ids[0]
    _enqueue_for_photo(tracked, test_user_id, JOB_TYPE_VLM_ANALYSIS)

    result = jr.photo_job_statuses(ids)

    assert len(result) == 120
    assert result[tracked]["vlm_status"] == "queued"
    assert all(result[i]["vlm_status"] == ABSENT for i in ids[1:])


def test_empty_list_returns_empty():
    assert jr.photo_job_statuses([]) == {}
