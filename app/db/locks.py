# Kaynak-scoped PostgreSQL advisory lock'lari (PR-2).
#
# AMAC: ayni is (job type + photo_id) iki worker tarafindan GERCEKTEN
# paralel islenmesin (bkz. jobs_repository.py basindaki analiz notu - reap
# canli ama heartbeat'i zamaninda yazamayan bir worker'i yanlislikla
# "olmus" sayabilir, is baska bir worker'a yeniden verilir).
#
# TASARIM KARARLARI (revizyon turlerinde tartisildi, TEKRAR ACMA):
#
#  - Advisory lock'lar VERITABANINA YERELDIR (cluster'a degil - LOCKTAG
#    icine MyDatabaseId gomulur). PhotoAI'nin veritabani adi "photoai_db"
#    (.env). Ayni cluster'daki farkli bir veritabaninda calisan baska bir
#    uygulama (ornegin IntelliumAI-Backend, farkli DB adi kullaniyorsa)
#    ile classid/objid CAKISMASI FIZIKSEL OLARAK MUMKUN DEGIL - bu yuzden
#    "classid registry" gibi bir koordinasyon GEREKMIYOR.
#
#  - Iki parametreli form pg_try_advisory_lock(classid, objid) kullanilir:
#    classid = is tipine ozgu sabit (asagida), objid = hashtext(photo_id).
#    face_pipeline ve vlm_analysis FARKLI tablolara yazdigi icin (Face/
#    Person/Cluster/Qdrant vs. PhotoAnalysis) birbirlerini BEKLEMEMELI.
#
#  - hashtext (int4, ~4.29e9 deger) 500K fotograf olceginde CAKISABILIR
#    (dogum gunu paradoksu ~29 beklenen cakisan cift) ama bu ZARARSIZDIR:
#    lock bir KIMLIK DOGRULAMASI degil, karsilikli-dislama araci - worker
#    hangi photo_id'yi islediginden BAGIMSIZ olarak bunu biliyor. Cakisma
#    sadece nadir/gereksiz bir bekleme/conflict yaratir, YANLIS SONUC
#    uretmez.
#
#  - SADECE pg_try_advisory_lock kullanilir (pg_advisory_lock ASLA
#    kullanilmaz - bkz. tests/test_advisory_locks.py::
#    test_no_blocking_advisory_lock_calls_anywhere_in_app, tum app/
#    agacini tarar). Boylece session-level advisory lock'un SAYACLI
#    olmasi (ayni session ayni kilidi iki kez alabilir, tek unlock
#    sayaci sifirlamaz) riski YAPISAL OLARAK ORTADAN KALKAR: kod
#    tabaninda bu kilidi alan TEK bir cagri yolu var, o da her zaman
#    esiyle (release_photo_lock) 1-1 eslenir.
#
#  - release_photo_lock, EScLESTIRILMIS pg_advisory_unlock cagirir (TUM
#    session'i degil - pg_advisory_unlock_all() KULLANILMAZ, cunku
#    gelecekte eklenecek ikinci bir kilit turu - ornegin Person/Cluster
#    centroid kilidi, bkz. B.5 analiz notu - varsa unlock_all o kilidi de
#    SESSIZCE dusurur). Donus False ise ("bende olmayan bir kilidi
#    birakmaya calistim") warning loglanir - bu, lock_acquired bayraginin
#    yanlis yonetildiginin ya da baska bir mantik hatasinin TEK
#    gozlemlenebilir sinyalidir.
#
#  - release exception verirse (baglanti zaten kopmus olabilir):
#    Session.invalidate() cagrilir (connection().invalidate() DEGIL -
#    ikincisi baglanti zaten kopmussa YENI/BASKA bir baglanti dondurup
#    MASUM bir baglantiyi imha edebilir). invalidate() cagrisindan sonra
#    exception YUTULUR (handler zaten isini bitirmis, release hatasi
#    yuzunden job'i fail etmek yanlis olur) AMA MUTLAKA logger.error ile
#    loglanir - invalidate edilmis bir baglanti, sizmis kilit ihtimalinin
#    en guclu tanisal sinyalidir; sessizce gecilirse worker/main.py'deki
#    lock-conflict escalasyon log'unun "neden" sorusuna cevap bulunamaz.

import logging
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("photoai.locks")

# Is tipine ozgu sabit namespace'ler. Anlamli olmalari GEREKMIYOR - sadece
# PhotoAI icinde face_pipeline/vlm_analysis'i ayirt eden, elle secilmis,
# sabit iki tam sayi olmalari yeterli.
PHOTOAI_LOCK_CLASS_FACE = 913001
PHOTOAI_LOCK_CLASS_VLM = 913002
# semantic_index isi - face/vlm ile AYNI photo_id icin bile birbirini
# BEKLEMEZ (farkli classid). Kod tabaninda advisory lock alan baska yer
# YOK (merge/label/reassign satir-seviyesi SELECT FOR UPDATE kullanir -
# bkz. identity_locks.py), bu deger 913001/913002 ile cakismaz.
PHOTOAI_LOCK_CLASS_SEMANTIC = 913003


def acquire_photo_lock(db: Session, classid: int, photo_id: uuid.UUID) -> bool:
    """pg_try_advisory_lock ile kaynagi (is tipi + photo_id) almaya calisir.

    BLOKE OLMAZ - aninda True/False doner. False donerse baska bir worker
    o an ayni (classid, photo_id) kaynagini isliyordur; cagiran taraf
    (photo_service.py) bunu LockConflict'e cevirir.

    hashtext(uuid) diye bir fonksiyon YOK (psycopg2 UUID'i "uuid" tipiyle
    gonderir) - bu yuzden hem str(photo_id) bind edilir HEM DE SQL'de
    acikca CAST(:photo_id AS text) yazilir (ikisi birden: str() tek
    basina yeterli olabilir ama CAST, ileride biri UUID objesini tekrar
    dogrudan gecerse SESSIZCE yanlis davranmak yerine ayni sekilde
    calismaya devam etmeyi garanti eder).
    """
    return bool(
        db.execute(
            text("SELECT pg_try_advisory_lock(:classid, hashtext(CAST(:photo_id AS text)))"),
            {"classid": classid, "photo_id": str(photo_id)},
        ).scalar()
    )


def acquire_content_lock(db: Session, classid: int, content_hash: str) -> bool:
    """acquire_photo_lock'in ICERIK-scoped esi (ingestion icin).

    Neden photo_id degil content_hash: ingestion'da korunmasi gereken kaynak
    "bu icerik"tir. Ayni fotografin iki FARKLI inbox dosyasi (iki farkli
    photo_id) ayni anda islenirse ikisi de "kayit yok" gorup INSERT etmeye
    calisirdi; photo_id-scoped bir kilit bunu ENGELLEMEZDI.

    TRANSACTION-SCOPED (pg_try_advisory_xact_lock) - acquire_photo_lock'un
    session-scoped'undan BILINCLI OLARAK FARKLI. Kilit, cagiran transaction
    commit ya da rollback olunca POSTGRES TARAFINDAN otomatik birakilir;
    eslesen bir unlock cagrisi YOKTUR (o yuzden release_content_lock de yok).

    NEDEN (gelistirme sirasinda YASANMIS bir hata):
      Session-scoped ilk denemede, hata yolundaki `db.rollback()` SQLAlchemy
      baglantiyi HAVUZA IADE ETTIGI icin, ardindan cagrilan unlock BASKA bir
      baglantida calisiyordu. Kilit, havuzdaki eski baglantida SIZILI
      kaliyordu ve o content_hash bir daha ASLA ingest edilemiyordu (testte
      'lock conflict' olarak yakalandi). Transaction-scoped kilitte bu hata
      sinifi YAPISAL OLARAK imkansiz: rollback zaten kilidi birakir.

    Kritik bolum tam olarak transaction'dir (SELECT ... -> INSERT -> COMMIT),
    dolayisiyla kilidin omru ile korunan bolgenin omru BIREBIR ortusur.

    Bloke ETMEZ (pg_try_* ailesi) - bkz. tests/test_advisory_locks.py.
    """
    return bool(
        db.execute(
            text("SELECT pg_try_advisory_xact_lock(:classid, hashtext(:content_hash))"),
            {"classid": classid, "content_hash": content_hash},
        ).scalar()
    )


def release_photo_lock(db: Session, classid: int, photo_id: uuid.UUID) -> None:
    """acquire_photo_lock ile ESLESTIRILMIS unlock. SADECE lock_acquired
    True oldugunda cagrilmali (photo_service.py'deki finally guard'i)."""
    try:
        released = bool(
            db.execute(
                text("SELECT pg_advisory_unlock(:classid, hashtext(CAST(:photo_id AS text)))"),
                {"classid": classid, "photo_id": str(photo_id)},
            ).scalar()
        )
        if not released:
            logger.warning(
                "pg_advisory_unlock False dondu: bende olmayan bir kilidi "
                "birakmaya calistim (classid=%s photo_id=%s) - lock_acquired "
                "bayragi yanlis yonetilmis olabilir, MANTIK HATASI SUPHESI.",
                classid, photo_id,
            )
    except Exception:
        logger.error(
            "release_photo_lock basarisiz (classid=%s photo_id=%s) - baglanti "
            "SIZMIS KILIT TASIYOR OLABILIR, pool'a fiziksel bag. iade edilmeden "
            "invalidate ediliyor.",
            classid, photo_id, exc_info=True,
        )
        db.invalidate()
