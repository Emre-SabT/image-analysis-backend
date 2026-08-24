"""Is kuyrugu worker'i.

Calistirma:
    JOB_TYPES=vlm_analysis  python -m app.worker.main
    JOB_TYPES=face_pipeline python -m app.worker.main

WORKER MODELDEN BAGIMSIZDIR: burada hicbir model/prompt/saglayici bilgisi
yoktur. VLM ayrintilari app/ai/dispatcher.py'nin, yuz hatti ayrintilari
app/services/face_service.py'nin arkasinda kalir; worker yalnizca
"is cek -> calistir -> sonuclandir" dongusunu yonetir.

TRANSACTION DUZENI (en kritik kisit):
    claim_next()   -> KISA transaction, hemen commit
    handler(...)   -> TRANSACTION DISINDA (VLM 8-20 sn surebilir)
    complete/fail  -> ayri, yine KISA transaction
"""

import logging
import os
import random
import signal
import sys
import threading
import time
import uuid

from app.core.settings import settings
from app.db.models import (
    JOB_TYPE_FACE_PIPELINE,
    JOB_TYPE_VLM_ANALYSIS,
    KNOWN_JOB_TYPES,
)
from app.db import jobs_repository as jobs
from app.services import photo_service
from app.services.photo_service import LockConflict

logger = logging.getLogger("photoai.worker")

# job.type -> isleyici. Isleyiciler photo_id alir, kendi DB oturumunu acar
# ve idempotenttir (bkz. photo_service.run_*_job).
HANDLERS = {
    JOB_TYPE_FACE_PIPELINE: photo_service.run_face_pipeline_job,
    JOB_TYPE_VLM_ANALYSIS: photo_service.run_vlm_analysis_job,
}


def parse_job_types(raw: str | None) -> list[str]:
    """JOB_TYPES'i ayristirir ve DOGRULAR - gecersizse SystemExit firlatir.

    FAIL-FAST, VARSAYILAN "TUMUNU TUKET" YOK: yanlis yapilandirilmis bir
    worker-face konteyneri sessizce VLM islerini de cekmeye baslarsa, tek
    GPU'yu paylasan VLM icin replica-bazli eszamanlilik kontrolu anlamsiz
    hale gelir (kaldirilan Semaphore(1)'in yerini alan mekanizma budur).
    Bu yuzden bos/tanimsiz/gecersiz JOB_TYPES ile worker BASLAMAZ.
    """
    if not raw or not raw.strip():
        raise SystemExit(
            "HATA: JOB_TYPES tanimsiz ya da bos. Worker hangi is tipini "
            "tuketecegini bilmeden BASLAMAZ (varsayilan 'tumunu tuket' "
            "davranisi bilincli olarak yoktur).\n"
            f"Gecerli tipler: {', '.join(sorted(KNOWN_JOB_TYPES))}\n"
            "Ornek: JOB_TYPES=vlm_analysis"
        )

    types = [t.strip() for t in raw.split(",") if t.strip()]
    if not types:
        raise SystemExit("HATA: JOB_TYPES yalnizca ayirac iceriyor, gecerli tip yok.")

    unknown = [t for t in types if t not in KNOWN_JOB_TYPES]
    if unknown:
        raise SystemExit(
            f"HATA: taninmayan is tipi: {', '.join(unknown)}\n"
            f"Gecerli tipler: {', '.join(sorted(KNOWN_JOB_TYPES))}"
        )

    if len(types) > 1:
        # Adim 5 olcumu: coklu tip claim_next'i ANY yoluna (Sort'lu, ~266x
        # daha yavas) dusurur. Calisir ama tavsiye edilmez.
        logger.warning(
            "JOB_TYPES birden fazla tip iceriyor (%s) - claim_next YAVAS "
            "ANY yoluna dusecek. Uretimde her worker TEK tip tuketmeli.",
            ", ".join(types),
        )
    return types


class _Heartbeat:
    """Uzun suren islerde locked_at'i tazeleyen arka plan thread'i.

    - Kendi KISA transaction'inda yazar; ana isi BLOKE ETMEZ.
    - complete/fail ile AYNI sahiplik kosulunu kullanir. Sahiplik kaybedilirse
      (reaper devralmis / baska worker almis) `lost` bayragi kalkar; worker
      isi birakir ve sonucu YAZMAYA CALISMAZ.
    """

    def __init__(self, job_id: uuid.UUID, worker_id: str, interval: int):
        self.job_id = job_id
        self.worker_id = worker_id
        self.interval = interval
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"hb-{str(job_id)[:8]}")

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                if not jobs.heartbeat(self.job_id, self.worker_id):
                    logger.warning(
                        "Heartbeat sahiplik kaybi: job=%s worker=%s - is birakiliyor",
                        self.job_id, self.worker_id,
                    )
                    self.lost.set()
                    return
            except Exception:
                logger.exception("Heartbeat hatasi: job=%s", self.job_id)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


class Worker:
    def __init__(self, job_types: list[str], worker_id: str | None = None):
        self.job_types = job_types
        self.worker_id = worker_id or f"{os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'host')}-{os.getpid()}"
        self._shutdown = threading.Event()
        self._last_reap = 0.0

    def request_shutdown(self, *_):
        """SIGTERM/SIGINT: mevcut isi BITIR, YENI is ALMA."""
        logger.info("Kapanma istegi alindi - mevcut is bitince duracak.")
        self._shutdown.set()

    def _maybe_reap(self):
        now = time.monotonic()
        if now - self._last_reap < settings.JOB_REAP_INTERVAL_SECONDS:
            return
        self._last_reap = now
        try:
            reaped = jobs.reap_stale(settings.JOB_STALE_TIMEOUT_SECONDS)
            if reaped:
                logger.info("reap_stale: %d is kuyruga geri konuldu", reaped)
        except Exception:
            logger.exception("reap_stale hatasi")

    def _process(self, job: jobs.ClaimedJob) -> None:
        handler = HANDLERS.get(job.type)
        if handler is None:
            jobs.fail(job.id, self.worker_id,
                      f"Isleyici tanimsiz: {job.type}", retry=False)
            return

        photo_id = job.payload.get("photo_id")
        if not photo_id:
            jobs.fail(job.id, self.worker_id,
                      "payload.photo_id eksik", retry=False)
            return

        # Heartbeat calisiyorken IS TRANSACTION DISINDA yurutulur.
        with _Heartbeat(job.id, self.worker_id,
                        settings.JOB_HEARTBEAT_SECONDS) as hb:
            try:
                handler(uuid.UUID(str(photo_id)))
            except LockConflict as exc:
                # GERCEK bir hata DEGIL: handler hic calismadi, baska bir
                # worker o an ayni kaynagi (is tipi + photo_id) isliyordu
                # (PR-2, pg_try_advisory_lock). fail() DEGIL, cezasiz
                # requeue - failure_count'a DOKUNMAZ (bkz. PR-1).
                if hb.lost.is_set():
                    logger.warning(
                        "job=%s sahiplik kaybedilmisti (lock-conflict sirasinda) "
                        "- requeue ATLANIYOR", job.id
                    )
                    return
                if job.attempts >= settings.JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS:
                    # attempts BURADA "kac kez claim edildi" anlaminda
                    # (failure_count DEGIL) - requeue_lock_conflict ona
                    # dokunmadigi icin bu sinirsiz surebilir. Esik asilinca
                    # OTOMATIK fail EDILMEZ (bkz. settings.py'deki gerekce -
                    # sizinti tipik olarak worker restart ile kendiliginden
                    # duzelir), sadece operator'un mudahale edebilmesi icin
                    # ESKALE EDILIR.
                    logger.error(
                        "job=%s type=%s photo_id=%s KILIT CAKISMASI ESIGI ASILDI "
                        "(attempts=%d >= %d) - SIZMIS ADVISORY LOCK SUPHESI. "
                        "Onerilen mudahale: ilgili worker process'lerini yeniden "
                        "baslat (pool'daki sizmis baglantiyi dusurur) ve/veya "
                        "pg_locks'ta locktype='advisory' satirlarini incele.",
                        job.id, job.type, photo_id,
                        job.attempts, settings.JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS,
                    )
                else:
                    logger.info("job=%s lock-conflict - cezasiz requeue: %s", job.id, exc)
                delay = settings.JOB_LOCK_CONFLICT_BACKOFF_SECONDS + random.uniform(0, 10)
                jobs.requeue_lock_conflict(job.id, self.worker_id, delay)
                return
            except Exception as exc:
                logger.exception("Is basarisiz: job=%s type=%s", job.id, job.type)
                if hb.lost.is_set():
                    logger.warning(
                        "job=%s sahiplik kaybedilmisti - sonuc YAZILMIYOR", job.id
                    )
                    return
                jobs.fail(job.id, self.worker_id, f"{type(exc).__name__}: {exc}")
                return

            if hb.lost.is_set():
                # Is bitti ama bu worker artik sahibi degil: complete()
                # cagirmak yeni sahibin sonucunu ezebilirdi.
                logger.warning(
                    "job=%s tamamlandi ama sahiplik kaybedilmis - complete ATLANIYOR",
                    job.id,
                )
                return

        if not jobs.complete(job.id, self.worker_id):
            logger.warning(
                "complete() 0 satir etkiledi (job=%s) - is baskasina gecmis olabilir",
                job.id,
            )

    def run(self) -> None:
        logger.info("Worker basladi: id=%s tipler=%s",
                    self.worker_id, ",".join(self.job_types))
        while not self._shutdown.is_set():
            self._maybe_reap()
            try:
                job = jobs.claim_next(self.worker_id, self.job_types)
            except Exception:
                logger.exception("claim_next hatasi")
                time.sleep(settings.JOB_POLL_INTERVAL_SECONDS)
                continue

            if job is None:
                # Kuyruk bos - kapanma istegini de dinleyerek bekle.
                self._shutdown.wait(settings.JOB_POLL_INTERVAL_SECONDS)
                continue

            logger.info("Is alindi: job=%s type=%s deneme=%d/%d",
                        job.id, job.type, job.attempts, job.max_attempts)
            self._process(job)

        logger.info("Worker temiz sekilde durdu: id=%s", self.worker_id)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # JOB_TYPES once ortam degiskeninden, yoksa settings'ten (ikisi de bos
    # olabilir - o zaman fail-fast devreye girer).
    raw = os.environ.get("JOB_TYPES", settings.JOB_TYPES)
    job_types = parse_job_types(raw)  # gecersizse SystemExit -> exit code != 0

    worker = Worker(job_types)
    signal.signal(signal.SIGTERM, worker.request_shutdown)
    signal.signal(signal.SIGINT, worker.request_shutdown)
    worker.run()


if __name__ == "__main__":
    sys.exit(main())
