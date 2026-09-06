"""Inbox -> DB ingestion: tamamlanmis bir inbox dosyasini kalici fotografa cevirir.

AKIS (tek dosya):

    inbox/{id}.jpg + inbox/{id}.jpg.meta.json
        -> SHA-256 (diskten, YENIDEN hesaplanir)
        -> content_hash advisory lock
        -> duplicate / yarim-tasima karari
        -> stored/{id}.jpg'e TASI
        -> Photo + PhotoExif + face_pipeline + vlm_analysis  (TEK COMMIT)
        -> meta yan-dosyasini sil

ISLEM SIRASI NEDEN BOYLE (en kritik tasarim karari):

  Dosya, DB commit'inden ONCE nihai yerine tasinir; meta yan-dosyasi ise
  commit'ten SONRA silinir. Yani meta'nin VARLIGI "bu dosyanin ingestion'i
  tamamlanmadi" isaretidir.

  Alternatif (once commit, sonra tasi) REDDEDILDI: commit is kayitlarini da
  gorunur yapar ve bir worker, dosya henuz stored/'a tasinmadan job'i claim
  edip "dosya yok" hatasiyla patlayabilirdi. Simdi is kaydi olustugu anda
  dosya ZATEN nihai yerindedir - yaris penceresi YOK.

  Bu siranin bedeli, "tasindi ama commit edilmedi" araligidir; onu da meta
  yan-dosyasi kapatir (bkz. recover_orphan_meta).

TEKILLIK: iki katman. (1) content_hash uzerinde PostgreSQL UNIQUE index -
ASIL garanti. (2) hashtext(content_hash) uzerinde advisory lock - ayni dosyayi
iki ingestion dongusunun ayni anda islemesini engeller. "Once SELECT sonra
INSERT" tek basina yeterli SAYILMAZ; UNIQUE ihlali ayrica yakalanir.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import jobs_repository, locks
from app.db.models import JOB_TYPE_FACE_PIPELINE, JOB_TYPE_VLM_ANALYSIS, Photo
from app.db.session import SessionLocal
from app.services import activity_log_service, photo_service
from app.services.photo_service import META_SUFFIX, PART_SUFFIX

# DIZINLER NITELIK UZERINDEN OKUNUR (photo_service.INBOX_DIR); modul
# seviyesinde "from app.services.photo_service import INBOX_DIR" ile
# BAGLANMAZ. Sebep: boyle bir bag, testlerin (ve ileride farkli bir depolama
# koku isteyen bir kurulumun) dizinleri yonlendirmesini imkansiz kilardi -
# erken baglanan isim, sonradan yapilan degisikligi gormez.

logger = logging.getLogger("photoai.ingestion")

# locks.py'deki 913001/913002/913003 ile CAKISMAZ. objid = hashtext(content_hash);
# kaynak burada "bu icerik" - photo_id degil, cunku ayni icerigi tasiyan IKI
# FARKLI inbox dosyasinin ayni anda islenmesini de engellemek istiyoruz.
PHOTOAI_LOCK_CLASS_INGEST = 913004

_HASH_CHUNK_BYTES = 1024 * 1024


class IngestOutcome:
    CREATED = "created"
    DUPLICATE = "duplicate"
    RECOVERED = "recovered"  # commit olmus ama dosyasi tasinmamis kayit tamamlandi
    SKIPPED = "skipped"      # kilit baskasinda / dosya artik yok
    FAILED = "failed"


@dataclass(frozen=True)
class IngestResult:
    outcome: str
    photo_id: uuid.UUID | None = None
    detail: str = ""


# --- yardimcilar -------------------------------------------------------


def hash_file(path: Path) -> str:
    """Dosyanin SHA-256'si - DISKTEN, parca parca.

    Meta'daki hash'e GUVENILMEZ ve yeniden hesaplanir: crash sonrasi
    kurtarmada tek dogruluk kaynagi diskteki baytlardir (meta dogru ama
    dosya yarim kalmis olabilir - .part deseni bunu onler ama savunma
    katmanini kaldirmak icin bir sebep degil)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _part_path_for(image_path: Path) -> Path:
    """receive_upload'in yazarken kullandigi gecici ad. Bir meta'nin
    "oksuz mu yoksa devam eden bir yukleme mi" oldugunu ayirt etmek icin."""
    return image_path.with_name(image_path.name + PART_SUFFIX)


def _move_to_failed(image_path: Path, reason: str) -> None:
    """Dosyayi ve meta'sini failed/ altina alir - SILMEZ. Kalici olarak
    islenemeyen dosya elle incelenebilsin diye (spec 12)."""
    try:
        photo_service.FAILED_DIR.mkdir(exist_ok=True)
        os.replace(image_path, photo_service.FAILED_DIR / image_path.name)
        meta = photo_service._meta_path_for(image_path)
        if meta.exists():
            os.replace(meta, photo_service.FAILED_DIR / meta.name)
        logger.error("[INGEST] Failed -> failed/ tasindi: %s (%s)", image_path.name, reason)
    except OSError:
        logger.exception("[INGEST] failed/ tasimasi da basarisiz: %s", image_path.name)


def list_ready_files() -> list[Path]:
    """Inbox'taki ISLENMEYE HAZIR goruntu dosyalari.

    .part (yaziliyor) ve .meta.json (yan-dosya) ATLANIR - yarim dosyanin
    islenmesi bu filtreyle yapisal olarak imkansiz olur."""
    if not photo_service.INBOX_DIR.exists():
        return []
    out = []
    for entry in sorted(photo_service.INBOX_DIR.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(PART_SUFFIX) or name.endswith(META_SUFFIX):
            continue
        out.append(entry)
    return out


# --- tek dosya ingestion ----------------------------------------------


def ingest_file(image_path: Path, db: Session | None = None) -> IngestResult:
    """Tek bir tamamlanmis inbox dosyasini sisteme alir.

    KENDI session'ini acar ve KENDI commit'ini yapar (worker handler'lari
    gibi) - cagiran dongu transaction yonetmez. `db` yalnizca testler icin.
    """
    own_session = db is None
    db = db or SessionLocal()
    try:
        return _ingest_file_inner(image_path, db)
    finally:
        if own_session:
            db.close()


def _ingest_file_inner(image_path: Path, db: Session) -> IngestResult:
    # Dosya ARADA KAYBOLDUYSA bu bir ariza DEGIL, kaybedilmis bir yaristir:
    # baska bir ingestion dongusu (uvicorn --workers, ya da --reload'un
    # gecici cift sureci) onu bizden once isleyip stored/'a tasimis olabilir.
    # Bu kontrol OLMADAN asagidaki "meta yok" dali devreye giriyor ve var
    # olmayan bir dosya icin _move_to_failed cagriliyordu: exception
    # traceback'i + 'failed' metriginin SAHTE artisi - operator icin gercek
    # arizadan ayirt edilemez gurultu.
    if not image_path.exists():
        return IngestResult(IngestOutcome.SKIPPED, detail="dosya artik yok (yaris kaybedildi)")

    meta = photo_service.read_inbox_meta(image_path)
    if meta is None:
        # META ZORUNLU DEGIL - eksikse SENTEZLENIR (bkz. _synthesize_meta).
        #
        # ONCEKI DAVRANIS bu dosyayi failed/'a aliyordu, gerekcesi
        # "yukleyiciyi bilemeyiz, sahte sahiplik uretmeyelim" idi. Gozden
        # gecirmede bu FAZLA KATI bulundu: on-premise'te inbox'a dogrudan
        # klasor kopyalamak mesru bir toplu ice aktarim yoludur ve o yolda
        # gelen SAGLAM fotograflar sessizce failed/'a dusuyordu.
        #
        # Sahte sahiplik de uretmiyoruz: kayit, ZATEN VAR OLAN ve bu is icin
        # tanimli sistem servis hesabina (settings.SYSTEM_USER_ID, is_active
        # False) yazilir - "bir insan yukledi" degil, "sistem ice aktardi"
        # demektir.
        try:
            meta = _synthesize_meta(image_path)
        except OSError as e:
            _move_to_failed(image_path, f"meta sentezlenemedi: {e}")
            return IngestResult(IngestOutcome.FAILED, detail=f"meta sentezlenemedi: {e}")

    logger.info("[INGEST] Found file: %s", image_path.name)

    try:
        content_hash = hash_file(image_path)
    except OSError as e:
        logger.warning("[INGEST] okunamadi (%s), sonraki turda tekrar denenecek: %s",
                       e, image_path.name)
        return IngestResult(IngestOutcome.SKIPPED, detail=f"okuma hatasi: {e}")
    logger.info("[INGEST] Hash calculated: %s hash=%s", image_path.name, content_hash[:12])

    # Ayni ICERIGI iki dongu/surec ayni anda islemesin. Bloke OLMAZ: kilit
    # baskasindaysa dosya bu turda atlanir, sonraki taramada tekrar gelir.
    #
    # Kilit TRANSACTION-SCOPED: commit/rollback ile Postgres tarafindan
    # otomatik birakilir, eslesen bir unlock cagrisi YOK (bkz. locks.py'deki
    # gerekce - session-scoped bir kilit, hata yolundaki rollback baglantiyi
    # havuza iade ettigi icin SIZAR).
    if not locks.acquire_content_lock(db, PHOTOAI_LOCK_CLASS_INGEST, content_hash):
        logger.info("[INGEST] kilit baskasinda, atlaniyor: %s", image_path.name)
        return IngestResult(IngestOutcome.SKIPPED, detail="lock conflict")

    try:
        existing = db.query(Photo).filter(Photo.content_hash == content_hash).first()
        if existing is not None:
            return _handle_existing(image_path, existing, content_hash)
        return _create_photo(image_path, meta, content_hash, db)
    except IntegrityError as e:
        # UNIQUE(content_hash) ihlali: kilit BASKA BIR SUREC tarafindan
        # tutulamayacagi icin buraya normalde dusulmez; yine de DB'nin
        # SON SOZU burasidir (spec 6: "once SELECT sonra INSERT tek basina
        # yeterli kabul edilmemeli"). Duplicate yoluna dusurulur.
        db.rollback()
        logger.info("[INGEST] Duplicate detected (DB UNIQUE): %s", image_path.name)
        existing = db.query(Photo).filter(Photo.content_hash == content_hash).first()
        if existing is not None:
            return _handle_existing(image_path, existing, content_hash)
        _move_to_failed(image_path, f"IntegrityError ama eslesen kayit yok: {e}")
        return IngestResult(IngestOutcome.FAILED, detail=str(e))
    except Exception as e:
        db.rollback()
        # BILINCLI: failed/'a TASIMIYORUZ. Gecici bir hata (DB kapali, disk
        # dolu) olabilir; dosya inbox'ta kalir ve sonraki uzlastirmada
        # yeniden denenir. Yalnizca KALICI olarak anlamsiz dosyalar
        # (meta yok / bozuk goruntu) failed/'a gider.
        logger.exception("[INGEST] Failed: %s", image_path.name)
        return IngestResult(IngestOutcome.FAILED, detail=str(e))


def _synthesize_meta(image_path: Path) -> dict:
    """Yan-dosyasi olmayan bir inbox dosyasi icin meta URETIR ve DISKE YAZAR.

    Neden diske de yaziyoruz (sadece bellekte tutmak yerine): meta'nin
    VARLIGI, crash kurtarmanin dayandigi "bu dosyanin ingestion'i bitmedi"
    isaretidir (bkz. modul basligi). Sentezlenen meta yazilmazsa, dosya
    stored/'a tasindiktan sonra commit'ten once yasanan bir crash GERIYE
    HICBIR IZ birakmaz ve dosya hicbir kaydin isaret etmedigi bir oksuz
    haline gelirdi.

    Yazim atomiktir (_write_inbox_meta: once .part, sonra rename).
    """
    from app.core.settings import settings

    meta = {
        "photo_id": str(uuid.uuid4()),
        "original_filename": image_path.name,
        "size_bytes": image_path.stat().st_size,
        "uploaded_by_user_id": settings.SYSTEM_USER_ID,
        "received_at": datetime.utcnow().isoformat(),
        # Kokeni ACIKCA isaretle - bu kayit bir HTTP yuklemesinden degil,
        # inbox'a dogrudan birakilmis bir dosyadan geldi.
        "source": "external_drop",
    }
    photo_service._write_inbox_meta(image_path, meta)
    logger.warning(
        "[INGEST] meta yan-dosyasi yok - DOGRUDAN BIRAKILMIS dosya olarak "
        "ice aktariliyor: %s (sahip: sistem servis hesabi)", image_path.name,
    )
    return meta


def _handle_existing(image_path: Path, existing: Photo, content_hash: str) -> IngestResult:
    """Ayni content_hash'e sahip bir kayit ZATEN VAR. Iki bambaska durum:

    (a) GERCEK DUPLICATE - kaydin dosyasi diskte duruyor. Inbox kopyasi
        bayt bayt ayni oldugu icin (SHA-256 ile dogrulandi) silinir. Veri
        KAYBI degil: ayni icerik stored/'da mevcut.

    (b) YARIM KALMIS TASIMA (crash senaryosu D) - kayit commit edilmis ama
        dosyasi nihai yerinde YOK. Bu, "kullanicinin ayni fotografi iki kez
        yuklemesi" DEGIL; bir onceki ingestion turunun tam ortasinda
        kapanmis olmasidir. Dosya nihai yerine tasinir, YENI KAYIT VE YENI
        IS OLUSTURULMAZ.
    """
    stored = Path(existing.storage_path)
    if stored.exists():
        # SILME GERI ALINAMAZ - once hedefe BAK. Inbox kopyasini silmenin
        # tek gerekcesi "ayni icerik zaten stored/'da duruyor" olmasi; bunu
        # varsaymak yerine dogruluyoruz. Boyut karsilastirmasi tek bir
        # stat() maliyetinde ve gercekci bozulma bicimini (kesilmis/eksik
        # dosya) yakalar; tam yeniden hash'leme her duplicate'te dosyanin
        # tamamini okumak demek olurdu.
        try:
            same_size = stored.stat().st_size == image_path.stat().st_size
        except OSError:
            same_size = False

        if not same_size:
            # Kayit ve dosya CELISIYOR. Inbox kopyasi tek saglam kopya
            # olabilir - SILMIYORUZ, elle inceleme icin failed/'a aliyoruz.
            _move_to_failed(
                image_path,
                f"content_hash mevcut photo_id={existing.id} ile eslesti ama "
                f"stored dosyasinin boyutu FARKLI ({stored}) - bozulma suphesi",
            )
            return IngestResult(
                IngestOutcome.FAILED, photo_id=existing.id,
                detail="stored dosya boyutu uyusmuyor",
            )

        photo_service._meta_path_for(image_path).unlink(missing_ok=True)
        image_path.unlink(missing_ok=True)
        logger.info("[INGEST] Duplicate detected: %s -> mevcut photo_id=%s",
                    image_path.name, existing.id)
        return IngestResult(IngestOutcome.DUPLICATE, photo_id=existing.id)

    stored.parent.mkdir(parents=True, exist_ok=True)
    os.replace(image_path, stored)
    photo_service._meta_path_for(image_path).unlink(missing_ok=True)
    logger.warning(
        "[INGEST] YARIM TASIMA TAMAMLANDI: photo_id=%s kaydi vardi ama dosyasi "
        "yoktu - inbox kopyasi %s konumuna tasindi. Yeni kayit/is OLUSTURULMADI.",
        existing.id, stored,
    )
    return IngestResult(IngestOutcome.RECOVERED, photo_id=existing.id)


def _create_photo(image_path: Path, meta: dict, content_hash: str, db: Session) -> IngestResult:
    photo_id = uuid.UUID(meta["photo_id"])
    filename = meta.get("original_filename") or image_path.name
    raw_user = meta.get("uploaded_by_user_id")
    user_id = uuid.UUID(raw_user) if raw_user else None
    size_bytes = int(meta.get("size_bytes") or image_path.stat().st_size)

    stored_path = photo_service.STORED_DIR / image_path.name
    photo_service.STORED_DIR.mkdir(exist_ok=True)

    # 1) DOSYAYI ONCE TASI (bkz. modul basindaki "islem sirasi" notu).
    #    Bu noktadan sonra crash olursa: dosya stored/'da, kayit YOK, meta
    #    hala inbox'ta -> recover_orphan_meta bunu yakalar.
    os.replace(image_path, stored_path)

    try:
        photo, exif = photo_service.build_photo_row(
            photo_id=photo_id,
            filename=filename,
            storage_path=stored_path,
            content_hash=content_hash,
            uploaded_by_user_id=user_id,
            size_bytes=size_bytes,
            exif_source_path=stored_path,
        )
        db.add(photo)
        db.add(exif)
        activity_log_service.log(db, user_id, "photo_upload", "photo", photo_id, filename)

        # IKI BAGIMSIZ IS - aralarinda sira/bagimlilik YOK (semantic_index
        # ucuncu bir hat olarak vlm_analysis BASARIYLA bitince dogar, bkz.
        # photo_service.run_vlm_analysis_job). Job sahibi olarak yukleyici
        # yoksa sabit sistem servis hesabi kullanilir.
        job_user = user_id or uuid.UUID(_system_user_id())
        jobs_repository.enqueue(db, JOB_TYPE_FACE_PIPELINE, {"photo_id": str(photo_id)}, job_user)
        jobs_repository.enqueue(db, JOB_TYPE_VLM_ANALYSIS, {"photo_id": str(photo_id)}, job_user)

        # 2) TEK ATOMIK COMMIT: foto + exif + activity + 2 job. Herhangi
        #    biri patlarsa HICBIRI kalici olmaz (spec 4.10).
        db.commit()
    except BaseException:
        # Commit olmadi: dosyayi inbox'a GERI al ki sonraki turda bastan
        # denensin (stored/'da oksuz dosya birakmayalim).
        try:
            os.replace(stored_path, image_path)
        except OSError:
            logger.exception("[INGEST] geri alma basarisiz: %s", stored_path)
        raise

    logger.info("[INGEST] Photo created: photo_id=%s file=%s", photo_id, filename)
    logger.info("[INGEST] Jobs created: face_pipeline + vlm_analysis (photo_id=%s)", photo_id)
    logger.info("[INGEST] DB commit successful: photo_id=%s", photo_id)

    # 3) Meta'yi EN SON sil - "ingestion bitti" isareti budur.
    photo_service._meta_path_for(image_path).unlink(missing_ok=True)
    logger.info("[INGEST] File moved to stored: %s", stored_path)
    return IngestResult(IngestOutcome.CREATED, photo_id=photo_id)


def _system_user_id() -> str:
    from app.core.settings import settings
    return settings.SYSTEM_USER_ID


# --- crash kurtarma / uzlastirma ---------------------------------------


def recover_orphan_meta() -> int:
    """Goruntu dosyasi OLMAYAN meta yan-dosyalarini uzlastirir.

    Bu durum tek bir sey demektir: `_create_photo` dosyayi stored/'a tasidi
    ama meta'yi silecek noktaya varamadi. Iki alt durum var:

      - DB'de kayit VAR  -> commit basariliydi, crash meta silinmeden once
                            oldu. Yapilacak: meta'yi sil.
      - DB'de kayit YOK  -> commit'ten ONCE crash oldu. Yapilacak: dosyayi
                            stored/'dan inbox'a GERI al; sonraki tur bastan
                            isler.

    Donus: uzlastirilan meta sayisi.
    """
    if not photo_service.INBOX_DIR.exists():
        return 0
    healed = 0
    db = SessionLocal()
    try:
        for meta_path in sorted(photo_service.INBOX_DIR.glob(f"*{META_SUFFIX}")):
            # HER META KENDI try'i icinde: tek bir bozuk yan-dosya (elle
            # olusturulmus, disk bozulmasi, eski surumden kalma) TUM
            # uzlastirmayi durdurmamali. Onceki halinde `uuid.UUID(photo_id)`
            # korumasizdi ve bozuk bir meta her turda ayni istisnayi
            # firlatiyordu; sirali listede ondan SONRA gelen oksuz metalar
            # BIR DAHA ASLA uzlastirilmiyordu - yani crash kurtarma kalici
            # olarak duruyordu.
            try:
                healed += _reconcile_one_meta(meta_path, db)
            except Exception:
                logger.exception(
                    "[INGEST] meta uzlastirilamadi, ATLANIYOR (digerleri devam "
                    "ediyor): %s", meta_path.name,
                )
    finally:
        db.close()
    return healed


def _reconcile_one_meta(meta_path: Path, db: Session) -> int:
    """Tek bir meta yan-dosyasini uzlastirir. Donus: 1 (islendi) / 0 (atlandi)."""
    image_path = photo_service.INBOX_DIR / meta_path.name[: -len(META_SUFFIX)]
    if image_path.exists():
        return 0  # normal bekleyen dosya, oksuz degil
    if _part_path_for(image_path).exists():
        # Yukleme SU AN suruyor (.part yaziliyor, meta zaten yazilmis).
        # Oksuz DEGIL - dokunma.
        return 0

    meta = photo_service.read_inbox_meta(image_path)
    photo_id = (meta or {}).get("photo_id")
    row = None
    if photo_id:
        try:
            row = db.query(Photo).filter(Photo.id == uuid.UUID(str(photo_id))).first()
        except ValueError:
            # photo_id bozuk: kaydi arayamayiz. Meta zaten kullanilamaz
            # durumda, asagida silinir.
            logger.error("[INGEST] meta'daki photo_id bozuk (%r): %s", photo_id, meta_path.name)

    if row is not None:
        meta_path.unlink(missing_ok=True)
        logger.info("[INGEST] oksuz meta temizlendi (kayit mevcut): photo_id=%s", photo_id)
        return 1

    stored_path = photo_service.STORED_DIR / image_path.name
    if stored_path.exists():
        os.replace(stored_path, image_path)
        logger.warning(
            "[INGEST] COMMIT ONCESI CRASH kurtarildi: %s stored/'dan inbox'a "
            "geri alindi, yeniden islenecek.", image_path.name,
        )
        return 1

    meta_path.unlink(missing_ok=True)
    logger.error(
        "[INGEST] oksuz meta ama dosya hicbir yerde yok: %s - meta silindi.",
        meta_path.name,
    )
    return 1


def sweep_stale_parts(max_age_seconds: int) -> int:
    """Uzun suredir dokunulmamis .part dosyalarini failed/'a alir.

    Bir .part yalnizca "istek ortasinda backend kapandi" durumunda kalir
    (receive_upload normal ve hatali yollarin IKISINDE de temizler). Yasli
    olma sarti onemli: SU AN yazilmakta olan bir yuklemeyi kesmemeliyiz.
    """
    import time

    if not photo_service.INBOX_DIR.exists():
        return 0
    now = time.time()
    swept = 0
    for part in sorted(photo_service.INBOX_DIR.glob(f"*{PART_SUFFIX}")):
        try:
            if now - part.stat().st_mtime < max_age_seconds:
                continue
            photo_service.FAILED_DIR.mkdir(exist_ok=True)
            os.replace(part, photo_service.FAILED_DIR / part.name)
            swept += 1
            logger.warning("[INGEST] olu .part failed/'a alindi: %s", part.name)
        except OSError:
            logger.exception("[INGEST] .part supurulemedi: %s", part.name)
    return swept


def inbox_stats() -> dict:
    """Gozlemlenebilirlik (spec 14): inbox'un o anki durumu.

    `orphan_meta` OPERATOR SINYALIDIR ve tek anlami olmalidir: "bir crash
    yasandi, uzlastirma bekleniyor". Bu yuzden goruntu dosyasi GERCEKTEN
    hicbir halde (ne hazir ne .part) bulunmayan metalar sayilir.

    Onceki hesap `metas - ready` idi ve YANLIS ALARM uretiyordu:
    receive_upload meta'yi nihai yeniden adlandirmadan ONCE yazdigi icin
    su an yuklenen her dosya bir "oksuz meta" gibi gorunuyordu - 12 es
    zamanli yuklemede operator `orphan_meta: 12` goruyordu.
    """
    if not photo_service.INBOX_DIR.exists():
        return {"ready": 0, "in_progress": 0, "orphan_meta": 0, "failed": 0}
    ready = len(list_ready_files())
    parts = sum(1 for _ in photo_service.INBOX_DIR.glob(f"*{PART_SUFFIX}"))
    orphan = 0
    for meta_path in photo_service.INBOX_DIR.glob(f"*{META_SUFFIX}"):
        image_path = photo_service.INBOX_DIR / meta_path.name[: -len(META_SUFFIX)]
        if not image_path.exists() and not _part_path_for(image_path).exists():
            orphan += 1
    failed = (
        sum(1 for p in photo_service.FAILED_DIR.iterdir() if p.is_file())
        if photo_service.FAILED_DIR.exists()
        else 0
    )
    return {"ready": ready, "in_progress": parts, "orphan_meta": orphan, "failed": failed}
