"""identity_locks.lock_identities testleri - GERCEK PostgreSQL'e karsi calisir.

Asil amac: deadlock ONLEME garantisini KANITLAMAK - iki thread AYNI iki
kimligi TERS sirada isterse (klasik deadlock senaryosu), ic siralama
sayesinde CEMBER BEKLEME olusmamali.
"""

import threading
import time
import uuid

import pytest
from sqlalchemy import text

from app.db import identity_locks
from app.db.session import SessionLocal


@pytest.fixture
def two_persons(test_user_id):
    """Kilitlenecek iki gercek Person satiri (throwaway)."""
    db = SessionLocal()
    ids = []
    try:
        for i in range(2):
            pid = uuid.uuid4()
            db.execute(
                text(
                    "INSERT INTO persons (id, display_name, face_count, created_by_user_id, created_at) "
                    "VALUES (:id, :n, 0, :u, now())"
                ),
                {"id": pid, "n": f"lock-test-{i}", "u": str(test_user_id)},
            )
            ids.append(pid)
        db.commit()
    finally:
        db.close()

    yield ids

    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM persons WHERE id::text = ANY(:ids)"), {"ids": [str(i) for i in ids]})
        db.commit()
    finally:
        db.close()


def test_lock_identities_acquires_and_releases(two_persons):
    a, b = two_persons
    db = SessionLocal()
    try:
        identity_locks.lock_identities(db, [("person", a), ("person", b)])
        db.commit()  # kilitler burada serbest kalir
    finally:
        db.close()

    # Baska bir session ayni satirlari kilitleyebiliyor mu (sizinti yok)?
    db2 = SessionLocal()
    try:
        identity_locks.lock_identities(db2, [("person", a), ("person", b)])
        db2.commit()
    finally:
        db2.close()


def test_opposite_order_requests_do_not_deadlock(two_persons):
    """KLASIK DEADLOCK SENARYOSU: thread 1 (A,B) sirasinda ister, thread 2
    (B,A) sirasinda ister - ic siralama OLMASAYDI cember bekleme olusurdu.
    Iki cagriyi da ARADA gecikmeyle (lock_identities'in KENDI dongusune
    degil, disaridan bir Barrier + ufak sleep ile gercek cakismayi
    zorlayarak) calistirip HER IKISININ DE (deadlock'a dusmeden) basariyla
    bittigini dogrular."""
    a, b = two_persons
    barrier = threading.Barrier(2, timeout=10)
    results = {}

    def worker(name, order):
        db = SessionLocal()
        try:
            barrier.wait()
            identity_locks.lock_identities(db, order, lock_timeout_ms=8000)
            time.sleep(0.05)  # kilit tutulurken cakismayi genislet
            db.commit()
            results[name] = "ok"
        except Exception as exc:  # pragma: no cover
            results[name] = f"error: {exc}"
        finally:
            db.close()

    t1 = threading.Thread(target=worker, args=("t1", [("person", a), ("person", b)]))
    t2 = threading.Thread(target=worker, args=("t2", [("person", b), ("person", a)]))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert results.get("t1") == "ok", results
    assert results.get("t2") == "ok", results


def test_lock_timeout_raises_instead_of_hanging_forever(two_persons):
    """Bir session kilidi TUTARKEN, digeri KISA bir lock_timeout ile
    isterse - sonsuza dek beklemek yerine zaman asimi hatasi almali."""
    a, _ = two_persons
    holder = SessionLocal()
    try:
        identity_locks.lock_identities(holder, [("person", a)])
        # holder kilidi tutuyor, commit ETMIYOR (bilerek acik birakiyoruz)

        waiter = SessionLocal()
        try:
            with pytest.raises(Exception):
                identity_locks.lock_identities(waiter, [("person", a)], lock_timeout_ms=300)
            waiter.rollback()
        finally:
            waiter.close()
    finally:
        holder.rollback()
        holder.close()


def test_empty_identities_is_a_no_op():
    db = SessionLocal()
    try:
        identity_locks.lock_identities(db, [])  # exception atmamali
        db.commit()
    finally:
        db.close()


def test_lock_timeout_raises_identity_lock_timeout(two_persons):
    """lock_timeout asilinca psycopg2'nin ham hatasi DEGIL, IdentityLockTimeout
    (Madde 4 - photo_service'te LockConflict'e cevrilecek TIP) firlamali."""
    a, _ = two_persons
    holder = SessionLocal()
    try:
        identity_locks.lock_identities(holder, [("person", a)])

        waiter = SessionLocal()
        try:
            with pytest.raises(identity_locks.IdentityLockTimeout):
                identity_locks.lock_identities(waiter, [("person", a)], lock_timeout_ms=300)
            waiter.rollback()
        finally:
            waiter.close()
    finally:
        holder.rollback()
        holder.close()
