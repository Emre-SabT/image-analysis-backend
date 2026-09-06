import os
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- TEST VERITABANI IZOLASYONU ---------------------------------------
#
# Testler URETIM veritabanini KULLANMAZ. Bu satirlar, app modulleri import
# EDILMEDEN ONCE calismak zorunda: `settings` bir modul-seviyesi singleton
# ve `app/db/session.py` engine'i import aninda kuruyor - env degiskeni
# sonradan degistirilirse hicbir etkisi olmaz.
#
# NEDEN GEREKLI (somut ariza): bu makinede uretim worker surecleri ve
# backend surekli calisiyor ve ayni veritabanina yaziyor. Paylasimli DB'de
# `test_worker_heartbeat.py` global canlilik durumuna assert ettigi icin
# 7 test HER KOSUDA kaliyordu - kod dogru oldugu halde. Ayni sinif kirlilik
# jobs/photos tablolarinda da mumkundu.
#
# Kacis kapisi: PHOTOAI_TEST_USE_MAIN_DB=1 -> eski (paylasimli) davranis.


def _read_database_url() -> str:
    """DATABASE_URL'i env'den, yoksa .env dosyasindan okur.
    (settings HENUZ import edilemez - bkz. yukaridaki not.)"""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("DATABASE_URL bulunamadi (env ya da .env)")


def _ensure_test_database() -> str:
    """Test veritabanini (yoksa) OLUSTURUR ve URL'ini doner.

    Ad, uretim veritabanindan `_test` sonekiyle turetilir - yanlislikla
    uretime baglanma ihtimalini gorsel olarak da azaltir.
    """
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    prod_url = _read_database_url()
    base, _, prod_db = prod_url.rpartition("/")
    test_db = os.environ.get("PHOTOAI_TEST_DB_NAME", f"{prod_db}_test")
    test_url = f"{base}/{test_db}"

    # Bakim veritabanina baglanip varligini kontrol et.
    admin = psycopg2.connect(f"{base}/postgres")
    admin.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        cur = admin.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db,))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{test_db}"')
            print(f"[conftest] test veritabani olusturuldu: {test_db}")
        cur.close()
    finally:
        admin.close()
    return test_url


if os.environ.get("PHOTOAI_TEST_USE_MAIN_DB") != "1":
    os.environ["DATABASE_URL"] = _ensure_test_database()

from app.db.session import SessionLocal, engine  # noqa: E402


def pytest_configure(config):
    """Test veritabani semasini goc'lerle guncel tutar (oturum basina bir kez).

    Alembic env.py `settings.DATABASE_URL` okur; yukarida env'i yonlendirdigimiz
    icin gocler TEST veritabanina uygulanir.
    """
    if os.environ.get("PHOTOAI_TEST_USE_MAIN_DB") == "1":
        return
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    command.upgrade(cfg, "head")

# jobs tablosunun TEMP kopyasini olusturan DDL.
#
# Neden TEMP ve neden yine "jobs" adiyla: PostgreSQL'de pg_temp,
# search_path'te public'ten ONCE gelir. Boylece test, uretimdeki SQL'i
# TEK KARAKTER DEGISTIRMEDEN (tablo adini yeniden yazmadan) calistirabilir
# ve yine de canli `public.jobs` verisine hic dokunmaz. Migration
# uygulanmis olsa da olmasa da test ayni sekilde calisir.
_TEMP_JOBS_DDL = """
CREATE TEMP TABLE jobs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type             VARCHAR(50)  NOT NULL,
    payload          JSONB        NOT NULL DEFAULT '{}',
    status           VARCHAR(20)  NOT NULL DEFAULT 'queued',
    priority         INTEGER      NOT NULL DEFAULT 0,
    user_id          UUID         NOT NULL,
    sequence_in_user BIGINT       NOT NULL,
    attempts         INTEGER      NOT NULL DEFAULT 0,
    failure_count    INTEGER      NOT NULL DEFAULT 0,
    max_attempts     INTEGER      NOT NULL DEFAULT 3,
    locked_by        VARCHAR(100),
    locked_at        TIMESTAMPTZ,
    run_after        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    error            TEXT,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ
);
CREATE INDEX ix_tmp_jobs_claim ON jobs (type, priority DESC, sequence_in_user ASC, created_at ASC)
    WHERE status = 'queued';
CREATE INDEX ix_tmp_jobs_reaper ON jobs (locked_at) WHERE status = 'running';
CREATE INDEX ix_tmp_jobs_user_queued ON jobs (user_id) WHERE status = 'queued';
"""

_SEED_ROWS = """
INSERT INTO jobs (type, status, priority, user_id, sequence_in_user, run_after, created_at)
SELECT
    CASE WHEN i %% 2 = 0 THEN 'vlm_analysis' ELSE 'face_pipeline' END,
    CASE WHEN i %% 100 < 80 THEN 'done'
         WHEN i %% 100 < 95 THEN 'queued'
         ELSE 'running' END,
    CASE WHEN i %% 50 = 0 THEN 10 ELSE 0 END,
    ('00000000-0000-0000-0000-' || lpad(((i %% 12) + 1)::text, 12, '0'))::uuid,
    (i / 12) + 1,
    now() - interval '1 hour',
    now() - (i || ' seconds')::interval
FROM generate_series(1, %d) AS i;
"""


@pytest.fixture
def jobs_conn():
    """TEMP `jobs` tablosu olan tek bir baglanti (canli veriye DOKUNMAZ)."""
    conn = engine.connect()
    try:
        conn.execute(text("SET search_path TO pg_temp, public"))
        # Havuzdan gelen baglantida onceki testin TEMP tablosu kalmis olabilir
        # (TEMP tablolar baglanti omru boyunca yasar, close() onlari dusurmez).
        conn.execute(text("DROP TABLE IF EXISTS pg_temp.jobs"))
        for stmt in _TEMP_JOBS_DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
        conn.commit()
        yield conn
    finally:
        try:
            conn.rollback()
            conn.execute(text("DROP TABLE IF EXISTS pg_temp.jobs"))
            conn.commit()
        except Exception:
            pass
        conn.close()


@pytest.fixture
def seeded_jobs_conn(jobs_conn):
    """EXPLAIN'in anlamli olmasi icin gercekci hacimde veri."""
    jobs_conn.execute(text(_SEED_ROWS % 20000))
    jobs_conn.execute(text("ANALYZE jobs"))
    jobs_conn.commit()
    return jobs_conn


@pytest.fixture(autouse=True)
def isolated_storage_dirs(tmp_path_factory, monkeypatch):
    """Her testi KENDI uploads/ kokune yonlendirir.

    NEDEN OTOMATIK (autouse) VE NEDEN SART: backend surecinde artik canli bir
    ingestion dongusu var (app/main.py lifespan) ve o, GERCEK uploads/inbox'i
    izliyor. Testler ayni dizine dosya birakirsa canli dongu onlari testin
    elinden KAPAR - testler rastgele "duplicate"/"dosya yok" ile patlar.
    (Bu, gelistirme sirasinda birebir yasandi.)

    photo_service.INBOX_DIR gibi modul-seviyesi Path'ler dogrudan
    degistirilir; ingestion_service bunlari NITELIK uzerinden okudugu icin
    (bkz. o dosyanin basindaki not) yonlendirme her iki modulde de gecerli
    olur.
    """
    from app.services import photo_service

    root = tmp_path_factory.mktemp("uploads")
    inbox, stored, failed = root / "inbox", root / "stored", root / "failed"
    for d in (inbox, stored, failed):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(photo_service, "INBOX_DIR", inbox)
    monkeypatch.setattr(photo_service, "STORED_DIR", stored)
    monkeypatch.setattr(photo_service, "FAILED_DIR", failed)
    yield root


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture
def test_user_id(db_session):
    """Testlerin FK kisitini saglamasi icin gecici bir kullanici."""
    from app.db.models import User

    uid = uuid.uuid4()
    db_session.add(
        User(
            id=uid,
            email=f"pytest-{uid}@photoai.internal",
            password_hash="x",
            display_name="pytest",
            role="editor",
        )
    )
    db_session.commit()
    yield uid
    db_session.execute(text("DELETE FROM jobs WHERE user_id = :u"), {"u": str(uid)})
    db_session.execute(
        text("DELETE FROM user_job_counters WHERE user_id = :u"), {"u": str(uid)}
    )
    # activity_log.actor_user_id SET NULL'dir (kullanici silinince PATLAMAZ) -
    # ama bu, testlerin urettigi satirlarin SESSIZCE "Bilinmeyen kullanici"
    # olarak GERCEK Son Etkinlik akisinda kalici kalmasi demek (bkz. Genel
    # Bakis'ta gorulen test kirliligi - "Test", "Ahmet", "P1"...). Testler
    # GERCEK DB'ye yazdigi icin (bkz. dosya basi yorumu - yalnizca `jobs`
    # sahte/TEMP) kendi urettikleri gunluk satirlarini da silmeli.
    db_session.execute(text("DELETE FROM activity_log WHERE actor_user_id = :u"), {"u": str(uid)})
    db_session.execute(text("DELETE FROM users WHERE id = :u"), {"u": str(uid)})
    db_session.commit()
