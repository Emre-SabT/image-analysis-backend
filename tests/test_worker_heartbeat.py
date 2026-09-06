"""worker_heartbeats - worker canlilik gozlemi (jobs_repository).

GERCEK PostgreSQL'e karsi calisir (conftest.py db_session gibi). `jobs`
TEMP kopyasi burada GEREKMEZ - bu tablo public.worker_heartbeats.
Testler kendi urettikleri satirlari worker_id onekiyle temizler.
"""

import pytest
from sqlalchemy import text

from app.db import jobs_repository as jr
from app.db.models import (
    JOB_TYPE_FACE_PIPELINE,
    JOB_TYPE_SEMANTIC_INDEX,
    JOB_TYPE_VLM_ANALYSIS,
    KNOWN_JOB_TYPES,
)

_PREFIX = "pytest-hb-"


@pytest.fixture(autouse=True)
def _clean_heartbeats(db_session):
    """Her testten once/sonra bu testlerin urettigi satirlari sil."""
    db_session.execute(
        text("DELETE FROM worker_heartbeats WHERE worker_id LIKE :p"),
        {"p": _PREFIX + "%"},
    )
    db_session.commit()
    yield
    db_session.execute(
        text("DELETE FROM worker_heartbeats WHERE worker_id LIKE :p"),
        {"p": _PREFIX + "%"},
    )
    db_session.commit()


def _age(db_session, worker_id: str, seconds: int) -> None:
    """last_seen'i geriye al - 'stale' / prune senaryolarini deterministik test et."""
    db_session.execute(
        text(
            "UPDATE worker_heartbeats "
            "SET last_seen = now() - (:s || ' seconds')::interval "
            "WHERE worker_id = :w"
        ),
        {"s": seconds, "w": worker_id},
    )
    db_session.commit()


def test_record_then_liveness_reports_ok():
    jr.record_heartbeat(_PREFIX + "sem-1", JOB_TYPE_SEMANTIC_INDEX)

    live = jr.worker_liveness(stale_seconds=45)
    # Tum bilinen tipler anahtar olarak var
    assert set(live) == set(KNOWN_JOB_TYPES)

    sem = live[JOB_TYPE_SEMANTIC_INDEX]
    assert sem["status"] == jr.WORKER_STATUS_OK
    assert sem["workers"] == 1
    assert sem["seconds_since"] is not None and sem["seconds_since"] < 45
    assert sem["last_seen"] is not None

    # Diger tipler icin worker yok -> down
    assert live[JOB_TYPE_VLM_ANALYSIS]["status"] == jr.WORKER_STATUS_DOWN
    assert live[JOB_TYPE_FACE_PIPELINE]["status"] == jr.WORKER_STATUS_DOWN


def test_stale_when_heartbeat_old(db_session):
    jr.record_heartbeat(_PREFIX + "sem-old", JOB_TYPE_SEMANTIC_INDEX)
    _age(db_session, _PREFIX + "sem-old", 120)

    sem = jr.worker_liveness(stale_seconds=45)[JOB_TYPE_SEMANTIC_INDEX]
    assert sem["status"] == jr.WORKER_STATUS_STALE
    assert sem["workers"] == 1
    assert sem["seconds_since"] >= 120


def test_down_when_no_rows():
    live = jr.worker_liveness(stale_seconds=45)
    assert all(v["status"] == jr.WORKER_STATUS_DOWN for v in live.values())
    assert all(v["workers"] == 0 for v in live.values())
    assert all(v["last_seen"] is None for v in live.values())


def test_upsert_refreshes_last_seen(db_session):
    jr.record_heartbeat(_PREFIX + "sem-2", JOB_TYPE_SEMANTIC_INDEX)
    _age(db_session, _PREFIX + "sem-2", 200)
    assert jr.worker_liveness(45)[JOB_TYPE_SEMANTIC_INDEX]["status"] == jr.WORKER_STATUS_STALE

    jr.record_heartbeat(_PREFIX + "sem-2", JOB_TYPE_SEMANTIC_INDEX)  # ayni worker_id
    sem = jr.worker_liveness(45)[JOB_TYPE_SEMANTIC_INDEX]
    assert sem["status"] == jr.WORKER_STATUS_OK
    assert sem["workers"] == 1  # UPSERT - yeni satir DEGIL


def test_multi_type_worker_counts_for_each_type():
    jr.record_heartbeat(_PREFIX + "multi", f"{JOB_TYPE_VLM_ANALYSIS},{JOB_TYPE_SEMANTIC_INDEX}")
    live = jr.worker_liveness(45)
    assert live[JOB_TYPE_VLM_ANALYSIS]["status"] == jr.WORKER_STATUS_OK
    assert live[JOB_TYPE_SEMANTIC_INDEX]["status"] == jr.WORKER_STATUS_OK
    assert live[JOB_TYPE_FACE_PIPELINE]["status"] == jr.WORKER_STATUS_DOWN


def test_freshest_wins_with_multiple_workers_same_type(db_session):
    jr.record_heartbeat(_PREFIX + "face-a", JOB_TYPE_FACE_PIPELINE)
    jr.record_heartbeat(_PREFIX + "face-b", JOB_TYPE_FACE_PIPELINE)
    _age(db_session, _PREFIX + "face-a", 300)  # biri olu

    face = jr.worker_liveness(45)[JOB_TYPE_FACE_PIPELINE]
    assert face["workers"] == 2
    assert face["status"] == jr.WORKER_STATUS_OK  # taze olan (face-b) kazanir
    assert face["seconds_since"] < 45


def test_clear_heartbeat_removes_row():
    jr.record_heartbeat(_PREFIX + "sem-3", JOB_TYPE_SEMANTIC_INDEX)
    assert jr.worker_liveness(45)[JOB_TYPE_SEMANTIC_INDEX]["status"] == jr.WORKER_STATUS_OK

    jr.clear_heartbeat(_PREFIX + "sem-3")
    assert jr.worker_liveness(45)[JOB_TYPE_SEMANTIC_INDEX]["status"] == jr.WORKER_STATUS_DOWN


def test_prune_stale_heartbeats(db_session):
    jr.record_heartbeat(_PREFIX + "dead", JOB_TYPE_SEMANTIC_INDEX)
    jr.record_heartbeat(_PREFIX + "alive", JOB_TYPE_SEMANTIC_INDEX)
    _age(db_session, _PREFIX + "dead", 3600)

    pruned = jr.prune_stale_heartbeats(older_than_seconds=300, session=db_session)
    assert pruned == 1

    rows = db_session.execute(
        text("SELECT worker_id FROM worker_heartbeats WHERE worker_id LIKE :p"),
        {"p": _PREFIX + "%"},
    ).scalars().all()
    assert rows == [_PREFIX + "alive"]
