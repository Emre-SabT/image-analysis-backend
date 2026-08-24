"""PostgreSQL tabanli is kuyrugu erisim katmani (FOR UPDATE SKIP LOCKED).

Bu projede daha once bir repository katmani yoktu (servisler `db: Session`
ile dogrudan calisiyordu). Kuyruk mantigi transaction sinirlarina cok
duyarli oldugu icin ayri bir modulde toplandi.

TRANSACTION KURALI (en kritik kisit): is (VLM 8-20 sn surebilir) HICBIR
zaman acik bir transaction icinde calistirilmaz. `claim_next` isi KISA bir
transaction'da kilitleyip commit eder; is o transaction KAPANDIKTAN sonra
calisir; `complete`/`fail` ayri ve yine kisa bir transaction acar.

AT-LEAST-ONCE: reaper nedeniyle ayni is birden fazla kez calisabilir.
Exactly-once hedeflenmiyor - sonucu yazan kod idempotent olmali.

TODO (bu PR'in KAPSAMI DISINDA, ayri gorev): BIRDEN FAZLA worker-face
sureci calistirildiginda Qdrant identity_pool'daki centroid guncellemeleri
gercek eszamanliliga maruz kalir; iki worker ayni kimligin centroid'ini ayni
anda guncellerse LOST UPDATE olusur. SELECT FOR UPDATE veya optimistic
locking ile korunmali. WORKER_FACE_PROCESSES > 1 yapilmadan ONCE
cozulmelidir.

AYNI IS IKI WORKER TARAFINDAN ISLENEBILIR MI (analiz notu): claim_next()
(FOR UPDATE SKIP LOCKED) claim ANINI korur, ama running durumundaki bir
isin SAHIPLIGI reap_stale()'in "locked_at eskimis mi" sezgisel kararina
dayanir - worker CANLI ama heartbeat'i (OS uykusu, DB kesintisi vb.
yuzunden) zamaninda yazilamazsa, is baska bir worker'a yanlislikla
yeniden verilebilir; bu araliktaki taraflar GERCEKTEN paralel calisir.
Bu PR (PR-1) bunun bir YAN ETKISINI (attempts/failure_count karisikligi -
asagida) duzeltiyor; asil onlem (photo-scoped advisory lock, PR-2) AYRI
bir degisiklikte geliyor.

ATTEMPTS / FAILURE_COUNT AYRIMI (PR-1): attempts HER claim'de artar (ilk
claim + her reaper-sonrasi reclaim), yani "kac kez GERCEKTEN denendi"
sorusuna guvenilir cevap DEGILDIR. failure_count ise SADECE fail()
gercekten cagrildiginda (handler gercekten calisip basarisiz olduğunda)
artar - max_attempts karsilastirmasi bu kolon uzerinden yapilir. Boylece
bir is yanlis-pozitif reap/lock-cakismasi yuzunden birden fazla kez
claim edilse bile, GERCEK ilk hatasinda erken/haksiz 'failed' olmaz.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import (
    JOB_TYPE_FACE_PIPELINE,
    JOB_TYPE_VLM_ANALYSIS,
    JOB_STATUS_DONE,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    Job,
)
from app.db.session import SessionLocal

# Sahiplik kosulu - complete/fail/heartbeat UCUNDE de ayni.
# Stale bir worker, isi devralan YENI worker'in sonucunu ezemesin diye.
_OWNERSHIP = "id = :job_id AND locked_by = :worker_id AND status = 'running'"

_BACKOFF_BASE_SECONDS = 10
_BACKOFF_MAX_SECONDS = 3600


@dataclass(frozen=True)
class ClaimedJob:
    """Kilitlenmis isin anlik goruntusu.

    ORM nesnesi DEGIL: claim transaction'i hemen kapandigi icin bir ORM
    Job dondurmek "detached instance" sorunlarina yol acardi. Bu duz veri
    nesnesi transaction'dan bagimsiz olarak guvenle tasinabilir.
    """

    id: uuid.UUID
    type: str
    payload: dict
    user_id: uuid.UUID
    attempts: int
    max_attempts: int


def build_claim_sql(job_types: Sequence[str]) -> str:
    """claim_next'in SQL'ini uretir - tek tip icin DUZ ESITLIK, coklu tip icin ANY.

    NEDEN BU DALLANMA VAR (Adim 5 olcumu, 200.000 satir / 29.999 queued):

      `type = ANY(:job_types)`  -> Postgres bunu ScalarArrayOpExpr olarak
      planlar. Dizinin birden fazla elemani icin sonuclari birlestirmek
      zorunda oldugundan indeksten gelen SIRALAMAYI GARANTI EDEMEZ ve
      araya bir `Sort` dugumu koyar. Bu, `LIMIT 1`'in "ilk indeks
      tuple'inda dur" avantajini tamamen yok eder:
          Bitmap + Sort, 15.999 satir taranir -> 2.449 buffer / 9,582 ms

      `type = :job_type`  (duz esitlik) -> lider kolon sabitlenir, geri
      kalan indeks kolonlari (priority DESC, sequence_in_user, created_at)
      ORDER BY ile birebir ortustugu icin siralama indeksten okunur,
      Sort dugumu OLUSMAZ:
          Index Scan using ix_jobs_claim  -> 5 buffer / 0,036 ms

    ~266x hiz farki, ama asil onemlisi OLCEKLENME YONU: duz esitlikte
    maliyet kuyruk derinliginden BAGIMSIZ, ANY yolunda kuyruk buyudukce
    dogrusal artar - bir kuyruk tablosu icin tam ters ozellik.

    Normal dagitimda her worker TEK tip tuketir (worker-vlm -> vlm_analysis,
    worker-face -> face_pipeline); JOB_TYPES fail-fast kuralinin varlik
    sebebi de budur. Coklu tip yolu DOGRU ama daha yavas bir yedektir.

    Bu dallanma tests/test_claim_sql_branch.py ile korunuyor - biri
    JOB_TYPES'i coklu tipe genisletirse regresyon testte yakalanir.
    """
    if len(job_types) == 1:
        type_predicate = "type = :job_type"
    else:
        type_predicate = "type = ANY(:job_types)"

    return f"""
        WITH next_job AS (
            SELECT id
            FROM jobs
            WHERE status = '{JOB_STATUS_QUEUED}'
              AND {type_predicate}
              AND run_after <= now()
            ORDER BY priority DESC, sequence_in_user ASC, created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE jobs j
        SET status     = '{JOB_STATUS_RUNNING}',
            locked_by  = :worker_id,
            locked_at  = now(),
            started_at = COALESCE(j.started_at, now()),
            attempts   = j.attempts + 1
        FROM next_job
        WHERE j.id = next_job.id
        RETURNING j.id, j.type, j.payload, j.user_id, j.attempts, j.max_attempts
    """


def build_claim_params(worker_id: str, job_types: Sequence[str]) -> dict:
    """build_claim_sql ile AYNI dallanmayi kullanmali - ikisi birlikte degisir."""
    if len(job_types) == 1:
        return {"worker_id": worker_id, "job_type": job_types[0]}
    return {"worker_id": worker_id, "job_types": list(job_types)}


# --- Uretici tarafi (cagiranin transaction'i icinde) --------------------


def enqueue(
    session: Session,
    job_type: str,
    payload: dict[str, Any],
    user_id: uuid.UUID,
    priority: int = 0,
) -> uuid.UUID:
    """Isi kuyruga ekler. CAGIRANIN ACIK SESSION'INI kullanir, commit ETMEZ.

    Boylece POST /photos, foto satirini ve IKI job satirini TEK atomik
    commit'te yazabilir: job insert patlarsa foto insert de geri alinir.

    sequence_in_user icin COUNT(*) BILINCLI OLARAK KULLANILMIYOR (yaris
    durumuna acik). Tek ifadelik atomik artirim kullaniliyor; satir yoksa
    ON CONFLICT ile olusturuluyor.
    """
    sequence = session.execute(
        text(
            """
            INSERT INTO user_job_counters (user_id, next_sequence)
            VALUES (:user_id, 1)
            ON CONFLICT (user_id) DO UPDATE
                SET next_sequence = user_job_counters.next_sequence + 1
            RETURNING next_sequence
            """
        ),
        {"user_id": str(user_id)},
    ).scalar_one()

    job = Job(
        id=uuid.uuid4(),
        type=job_type,
        payload=payload,
        status=JOB_STATUS_QUEUED,
        priority=priority,
        user_id=user_id,
        sequence_in_user=sequence,
        attempts=0,
        max_attempts=3,
    )
    session.add(job)
    session.flush()  # id'yi al; commit CAGIRANA ait
    return job.id


# --- Tuketici tarafi (worker - kendi KISA transaction'lari) -------------


def claim_next(worker_id: str, job_types: Sequence[str]) -> ClaimedJob | None:
    """Bir isi kilitleyip 'running' isaretler ve transaction'i HEMEN kapatir.

    Is, bu fonksiyon dondukten SONRA (transaction disinda) calistirilir.

    attempts CLAIM aninda artar (fail aninda degil): worker cokup is reaper
    tarafindan geri alinsa bile deneme sayilir, boylece sonsuz reap dongusu
    olusamaz.
    """
    if not job_types:
        raise ValueError("job_types bos olamaz")

    db = SessionLocal()
    try:
        row = db.execute(
            text(build_claim_sql(job_types)),
            build_claim_params(worker_id, job_types),
        ).first()
        db.commit()
    finally:
        db.close()

    if row is None:
        return None
    return ClaimedJob(
        id=row.id,
        type=row.type,
        payload=row.payload or {},
        user_id=row.user_id,
        attempts=row.attempts,
        max_attempts=row.max_attempts,
    )


def complete(job_id: uuid.UUID, worker_id: str) -> bool:
    """Isi 'done' isaretler. SONUC ALMAZ (jobs bir koordinasyon tablosu).

    Sahiplik kosulu saglanmazsa (reaper devralmis, baska worker almis)
    False doner - HATA FIRLATMAZ.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                f"""
                UPDATE jobs
                SET status = '{JOB_STATUS_DONE}', finished_at = now(),
                    locked_by = NULL, locked_at = NULL, error = NULL
                WHERE {_OWNERSHIP}
                """
            ),
            {"job_id": str(job_id), "worker_id": worker_id},
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def fail(job_id: uuid.UUID, worker_id: str, error: str, retry: bool = True) -> bool:
    """Isi basarisiz isaretler; retry ve deneme hakki varsa kuyruga geri koyar.

    max_attempts karsilastirmasi failure_count uzerinden yapilir, attempts
    UZERINDEN DEGIL (bkz. migration f1a2b3c4d5e6 ve modeldeki yorum):
    attempts claim SAYISIDIR (reaper'in yanlis-pozitif reclaim'i dahil),
    failure_count ise SADECE bu fonksiyonun gercekten cagrildigi (yani
    handler'in gercekten calisip basarisiz oldugu) durumlarda artar. Bu
    UPDATE, "failure_count + 1"i her CASE/backoff ifadesinde tutarli
    sekilde kullanir - Postgres tek bir UPDATE icinde SET ifadelerini
    satirin ESKI (guncellemeden onceki) degerleriyle degerlendirir, yani
    hepsi ayni "yeni deger" uzerinden hesaplanmis olur.

    Geri cekilme (backoff) usteldir ve bir ust sinira kadar buyur.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                f"""
                UPDATE jobs
                SET failure_count = failure_count + 1,
                    status = CASE
                        WHEN :retry AND (failure_count + 1) < max_attempts THEN '{JOB_STATUS_QUEUED}'
                        ELSE '{JOB_STATUS_FAILED}'
                    END,
                    error = :error,
                    locked_by = NULL,
                    locked_at = NULL,
                    run_after = CASE
                        WHEN :retry AND (failure_count + 1) < max_attempts
                            THEN now() + (LEAST(
                                :backoff_base * POWER(2, failure_count),
                                :backoff_max
                            ) || ' seconds')::interval
                        ELSE run_after
                    END,
                    finished_at = CASE
                        WHEN :retry AND (failure_count + 1) < max_attempts THEN NULL
                        ELSE now()
                    END
                WHERE {_OWNERSHIP}
                """
            ),
            {
                "job_id": str(job_id),
                "worker_id": worker_id,
                "error": error[:5000],
                "retry": retry,
                "backoff_base": _BACKOFF_BASE_SECONDS,
                "backoff_max": _BACKOFF_MAX_SECONDS,
            },
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def requeue_lock_conflict(job_id: uuid.UUID, worker_id: str, delay_seconds: float) -> bool:
    """Kaynak-kilidi cakismasi (PR-2, photo-scoped advisory lock) sonucu isi
    CEZASIZ kuyruga geri koyar.

    fail()'DEN BILINCLI OLARAK AYRI: bir baska worker'in ayni photo_id/is
    tipini O ANDA isliyor olmasi GERCEK bir hata degildir - handler hic
    calismamistir. Bu yuzden failure_count'a DOKUNMAZ (retry butcesini
    tuketmez); attempts zaten claim_next() tarafindan artmis olacaktir
    (bu fonksiyon onu da degistirmez).

    delay_seconds CAGIRANDAN gelir (worker/main.py, JOB_LOCK_CONFLICT_
    BACKOFF_SECONDS + jitter) - JOB_STALE_TIMEOUT_SECONDS'tan bagimsiz,
    cok daha kisa bir bekleme suresi olmasi beklenir (bu bir "worker
    oldu mu" karari degil, "kaynak o an mesguldu" bilgisi).
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                f"""
                UPDATE jobs
                SET status = '{JOB_STATUS_QUEUED}',
                    locked_by = NULL,
                    locked_at = NULL,
                    run_after = now() + (:delay || ' seconds')::interval
                WHERE {_OWNERSHIP}
                """
            ),
            {"job_id": str(job_id), "worker_id": worker_id, "delay": delay_seconds},
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def heartbeat(job_id: uuid.UUID, worker_id: str) -> bool:
    """locked_at'i tazeler ki uzun suren is reaper tarafindan stale sayilmasin.

    complete/fail ile AYNI sahiplik kosulunu kullanir. False donerse is
    artik bu worker'a ait DEGILDIR (reaper devralmis ya da baska worker
    almistir) - worker mevcut isi birakmali, sonucu YAZMAYA CALISMAMALIDIR.

    Kendi KISA transaction'inda calisir; ana isi bloke etmez.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text(f"UPDATE jobs SET locked_at = now() WHERE {_OWNERSHIP}"),
            {"job_id": str(job_id), "worker_id": worker_id},
        )
        db.commit()
        return result.rowcount == 1
    finally:
        db.close()


def reap_stale(timeout_seconds: int) -> int:
    """locked_at'i timeout'tan eski olan 'running' isleri kuyruga geri koyar."""
    db = SessionLocal()
    try:
        result = db.execute(
            text(
                f"""
                UPDATE jobs
                SET status = '{JOB_STATUS_QUEUED}', locked_by = NULL, locked_at = NULL
                WHERE status = '{JOB_STATUS_RUNNING}'
                  AND locked_at < now() - (:timeout || ' seconds')::interval
                """
            ),
            {"timeout": timeout_seconds},
        )
        db.commit()
        return result.rowcount
    finally:
        db.close()


# --- Sorgular ----------------------------------------------------------


def get_status(job_id: uuid.UUID, session: Session | None = None) -> dict | None:
    own = session is None
    db = session or SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT id, type, status, priority, attempts, failure_count, max_attempts,
                       error, created_at, started_at, finished_at, run_after
                FROM jobs WHERE id = :job_id
                """
            ),
            {"job_id": str(job_id)},
        ).mappings().first()
        return dict(row) if row else None
    finally:
        if own:
            db.close()


def count_queued_for_user(user_id: uuid.UUID, session: Session | None = None) -> int:
    own = session is None
    db = session or SessionLocal()
    try:
        return db.execute(
            text(
                f"SELECT count(*) FROM jobs "
                f"WHERE user_id = :user_id AND status = '{JOB_STATUS_QUEUED}'"
            ),
            {"user_id": str(user_id)},
        ).scalar_one()
    finally:
        if own:
            db.close()


def queue_status_by_type(
    user_id: uuid.UUID, eta_sample_size: int, session: Session | None = None
) -> dict:
    """Job tipi BAZINDA ayri sayim + ayri tahmini sure.

    face (~0,13 sn) ve vlm (~8-20 sn) sureleri buyuklukce farkli oldugundan
    KARISIK TEK ORTALAMA anlamsiz olurdu. Tahmini sure son N tamamlanmis
    isin OLCULEN ortalamasindan gelir - sabit bir tahmin degeri yok.
    """
    own = session is None
    db = session or SessionLocal()
    try:
        rows = (
            db.execute(
                text(
                    f"""
                    SELECT type, count(*) AS queued
                    FROM jobs
                    WHERE user_id = :user_id AND status = '{JOB_STATUS_QUEUED}'
                    GROUP BY type
                    """
                ),
                {"user_id": str(user_id)},
            )
            .mappings()
            .all()
        )
        queued_by_type = {r["type"]: r["queued"] for r in rows}

        out: dict[str, dict] = {}
        for job_type, queued in queued_by_type.items():
            avg_seconds = db.execute(
                text(
                    f"""
                    SELECT avg(EXTRACT(EPOCH FROM (finished_at - started_at)))
                    FROM (
                        SELECT started_at, finished_at FROM jobs
                        WHERE type = :job_type AND status = '{JOB_STATUS_DONE}'
                          AND started_at IS NOT NULL AND finished_at IS NOT NULL
                        ORDER BY finished_at DESC LIMIT :sample
                    ) recent
                    """
                ),
                {"job_type": job_type, "sample": eta_sample_size},
            ).scalar()
            out[job_type] = {
                "queued": queued,
                "avg_seconds": round(float(avg_seconds), 2) if avg_seconds else None,
                "eta_seconds": (
                    round(float(avg_seconds) * queued, 2) if avg_seconds else None
                ),
            }
        return out
    finally:
        if own:
            db.close()


# --- Fotograf bazli durum (GET /photos/{id}/status icin) ----------------

JOB_STATUS_ABSENT = "absent"


def photo_job_statuses(
    photo_ids: Sequence[str], session: Session | None = None
) -> dict[str, dict[str, str]]:
    """Verilen fotograflarin face/vlm is durumlarini tek sorguda dondurur.

    Donus: {photo_id: {"face_status": ..., "vlm_status": ...}}

    'absent': o fotograf icin O TIPTE hic job satiri YOK. Iki mesru
    sebebi var ve ikisi de "islenecek bir sey yok" anlamina gelir:
      1. Kuyruk sisteminden ONCE yuklenmis eski fotograflar (canli
         veride 446 adet) - zaten islenmis durumdalar.
      2. Duplicate yukleme - ayni icerik daha once islendigi icin
         bilincli olarak job ACILMAZ (bkz. routers/photos.upload_photo).
    Frontend 'absent'i "yerlesmis" kabul eder: ne spinner gosterir ne
    de sorgulamaya devam eder. (Duplicate bilgisi kullaniciya ZATEN
    POST /photos yanitindaki `duplicate: true` ile bildiriliyor -
    burada tekrar edilmesine gerek yok.)

    payload->>'photo_id' uzerindeki ix_jobs_photo_id indeksini kullanir
    (migration e8b4d2f1a6c3) - bu uc polling ile surekli cagrildigi ve
    jobs tablosu sinirsiz buyudugu icin indeks sart.
    """
    ids = [str(p) for p in photo_ids]
    out: dict[str, dict[str, str]] = {
        pid: {"face_status": JOB_STATUS_ABSENT, "vlm_status": JOB_STATUS_ABSENT}
        for pid in ids
    }
    if not ids:
        return out

    own = session is None
    db = session or SessionLocal()
    try:
        rows = db.execute(
            text(
                """
                SELECT payload->>'photo_id' AS photo_id, type, status, created_at
                FROM jobs
                WHERE payload->>'photo_id' = ANY(:ids)
                ORDER BY created_at ASC
                """
            ),
            {"ids": ids},
        ).mappings().all()
    finally:
        if own:
            db.close()

    key_by_type = {
        JOB_TYPE_FACE_PIPELINE: "face_status",
        JOB_TYPE_VLM_ANALYSIS: "vlm_status",
    }
    for row in rows:
        pid = row["photo_id"]
        key = key_by_type.get(row["type"])
        if pid in out and key:
            # created_at ASC siralamasi sayesinde ayni (foto, tip) icin
            # birden fazla satir olursa EN YENISI kazanir.
            out[pid][key] = row["status"]
    return out
