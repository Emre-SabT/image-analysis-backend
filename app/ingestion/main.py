"""Inbox izleme dongusu.

Backend surecinin ICINDE bir arka plan thread'i olarak calisir (bkz.
app/main.py lifespan). Boylece "backend yeniden baslayinca ingestion da
otomatik baslar" garantisi, ayri bir pencere/servis yonetmeye gerek
kalmadan saglanir - on-premise .bat tabanli kurulumumuza uyan bicim budur.

CALISMA SEKLI - iki katman:

  1. HIZLI YOKLAMA (INGESTION_POLL_INTERVAL_SECONDS, ~1 sn): inbox'ta hazir
     dosya var mi diye bakar. Yeni fotografin isleme girme gecikmesi budur.

  2. TAM UZLASTIRMA (INGESTION_RECONCILE_INTERVAL_SECONDS + ACILISTA BIR KEZ):
     oksuz meta yan-dosyalarini ve olu .part dosyalarini ele alir. Crash
     kurtarmanin gerceklestigi yer burasidir.

FILESYSTEM EVENT (watchdog) BILINCLI OLARAK KULLANILMIYOR: yeni bir
bagimlilik getirir, Windows'ta ag/paylasimli dizinlerde guvenilmezdir ve
"sadece event'e guvenme" geregi yuzunden periyodik tarama ZATEN sart. Yerel
bir dizinde saniyede bir os.scandir, olculebilir bir maliyet degil.

COK-SURECLI GUVENLIK: uvicorn `--workers N` ile calistirilirsa N adet
ingestion dongusu olusur. Bu GUVENLIDIR - her dosya, icerik-scoped advisory
kilit altinda islenir (bkz. ingestion_service); kilidi alamayan dongu dosyayi
atlar. Islem tekrari ya da yinelenen is kaydi olusmaz.
"""

from __future__ import annotations

import logging
import threading
import time

from app.core.settings import settings
from app.services import ingestion_service, photo_service
from app.services.ingestion_service import IngestOutcome

logger = logging.getLogger("photoai.ingestion")


class IngestionLoop:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_reconcile = 0.0
        self.stats = {"created": 0, "duplicate": 0, "recovered": 0, "failed": 0}
        self.last_reconcile_at: float | None = None

    # --- yasam dongusu ---

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ingestion", daemon=True)
        self._thread.start()
        logger.info(
            "[INGEST] dongu basladi (poll=%ss reconcile=%ss inbox=%s)",
            settings.INGESTION_POLL_INTERVAL_SECONDS,
            settings.INGESTION_RECONCILE_INTERVAL_SECONDS,
            photo_service.INBOX_DIR,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("[INGEST] dongu durdu. Toplam: %s", self.stats)

    # --- ic dongu ---

    def _run(self) -> None:
        # ACILISTA UZLASTIRMA: onceki calisma bir crash ile bitmis olabilir.
        # Hizli yoklamadan ONCE kosar ki yarim kalmis islemler once duzelsin.
        self._reconcile()
        while not self._stop.is_set():
            try:
                processed = self._drain_once()
            except Exception:
                logger.exception("[INGEST] dongu turu basarisiz - devam ediliyor")
                processed = 0

            now = time.monotonic()
            if now - self._last_reconcile >= settings.INGESTION_RECONCILE_INTERVAL_SECONDS:
                self._reconcile()

            # Is varken beklemeden devam et (100 fotograflik bir tur
            # saniyede bir dosyayla degil, ardisik olarak isler).
            if processed == 0:
                self._stop.wait(settings.INGESTION_POLL_INTERVAL_SECONDS)

    def _drain_once(self) -> int:
        """Hazir dosyalari isler. Donus: islenen dosya sayisi.

        BATCH_LIMIT ile sinirli: cok buyuk bir inbox'ta dongunun kapanma
        istegine ve uzlastirmaya donebilmesi icin.
        """
        files = ingestion_service.list_ready_files()[: settings.INGESTION_BATCH_LIMIT]
        count = 0
        for path in files:
            if self._stop.is_set():
                break
            result = ingestion_service.ingest_file(path)
            if result.outcome in self.stats:
                self.stats[result.outcome] += 1
            # SKIPPED sayilmaz: kilit cakismasi/gecici okuma hatasi, dosya
            # sonraki turda tekrar denenecek.
            if result.outcome != IngestOutcome.SKIPPED:
                count += 1
        return count

    def _reconcile(self) -> None:
        self._last_reconcile = time.monotonic()
        try:
            healed = ingestion_service.recover_orphan_meta()
            swept = ingestion_service.sweep_stale_parts(settings.INGESTION_STALE_PART_SECONDS)
            if healed or swept:
                logger.info(
                    "[INGEST] uzlastirma: %d oksuz meta duzeltildi, %d olu .part supuruldu",
                    healed, swept,
                )
        except Exception:
            logger.exception("[INGEST] uzlastirma basarisiz")
        self.last_reconcile_at = time.time()


# Surec basina TEK dongu.
loop = IngestionLoop()


def main() -> None:
    """Dongunun AYRI bir surec olarak calistirilabilmesi icin giris noktasi:

        python -m app.ingestion.main

    Normal kurulumda GEREKMEZ (backend lifespan'i zaten baslatiyor). Ileride
    ingestion'i API'den ayirmak istenirse INGESTION_ENABLED=false ile backend
    icindeki dongu kapatilip bu komut ayri bir surec olarak kosulabilir.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    loop.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        loop.stop()


if __name__ == "__main__":
    main()
