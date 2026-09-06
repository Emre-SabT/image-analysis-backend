"""Inbox -> ingestion -> is kuyrugu hattinin davranis testleri.

Kapsam (kullanicinin kabul kriterlerine gore):
  TEST 1  worker, TUM fotograflarin yuklenmesini BEKLEMEDEN ilkini alabiliyor
  TEST 4  ingestion crash/restart - oksuz meta uzlastirmasi
  TEST 6  ayni content_hash'li iki dosya -> TEK photo, is kaydi ikilenmiyor
  TEST 8  worker inbox'u DEGIL, DB kuyrugunu tuketiyor
  + .part dosyalarinin ASLA islenmemesi (yarim dosya korumasi)
"""

import io
import time
import os
import uuid
from pathlib import Path

from PIL import Image
from sqlalchemy import text

from app.db import jobs_repository as jr
from app.db.models import JOB_TYPE_FACE_PIPELINE, Photo
from app.db.session import SessionLocal
from app.services import ingestion_service, photo_service
from app.services.ingestion_service import IngestOutcome

_PREFIX = "pipeline-test-"


def _random_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (8, 8))
    img.putdata([tuple(os.urandom(3)) for _ in range(64)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _place_in_inbox(user_id, data: bytes | None = None, name: str | None = None) -> tuple[Path, uuid.UUID]:
    """receive_upload'in urettigi sekli olusturur (goruntu + yan-dosya)."""
    photo_id = uuid.uuid4()
    data = data if data is not None else _random_jpeg_bytes()
    image_path = photo_service.INBOX_DIR / f"{photo_id}.jpg"
    image_path.write_bytes(data)
    photo_service._write_inbox_meta(
        image_path,
        {
            "photo_id": str(photo_id),
            "original_filename": name or f"{_PREFIX}{photo_id}.jpg",
            "content_hash": ingestion_service.hash_file(image_path),
            "size_bytes": len(data),
            "uploaded_by_user_id": str(user_id),
            "received_at": "2026-09-04T00:00:00",
        },
    )
    return image_path, photo_id


def _purge(photo_ids, user_id):
    db = SessionLocal()
    try:
        for pid in photo_ids:
            db.execute(text("DELETE FROM jobs WHERE payload->>'photo_id' = :p"), {"p": str(pid)})
            db.execute(text("DELETE FROM activity_log WHERE target_id=:p"), {"p": str(pid)})
            db.execute(text("DELETE FROM photo_exif WHERE photo_id=:p"), {"p": str(pid)})
            db.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(pid)})
        db.commit()
    finally:
        db.close()
    for pid in photo_ids:
        for p in (photo_service.STORED_DIR / f"{pid}.jpg", photo_service.INBOX_DIR / f"{pid}.jpg"):
            p.unlink(missing_ok=True)
            photo_service._meta_path_for(p).unlink(missing_ok=True)


def _drain_queue(user_id):
    """Bu kullaniciya ait queued isleri kuyruktan cikarir - testler
    birbirinin kuyrugunu gormesin."""
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM jobs WHERE user_id = :u"), {"u": str(user_id)})
        db.commit()
    finally:
        db.close()


# --- TEST 1 -----------------------------------------------------------


def test_worker_can_start_before_all_uploads_are_ingested(test_user_id):
    """TEST 1 - EN KRITIK GARANTI.

    20 dosya inbox'a birakilir (kullanici hala yukluyor), ingestion bunlarin
    yalnizca ILK UCUNU isler. Bu noktada worker'in alabilecegi isin kuyrukta
    HAZIR olmasi gerekir: sistem 'once hepsi gelsin, sonra basla' MANTIGIYLA
    CALISMAMALI.

    Bu birim testi kuyrugun durumunu dogrular. Hattin ucundan uca gercek
    davranisi (calisan backend + gercek worker surecleriyle 100 fotograf)
    ayrica olculdu; orada ilk isler, upload'larin %15'i tamamlanmisken
    bitmisti.
    """
    _drain_queue(test_user_id)
    placed = [_place_in_inbox(test_user_id) for _ in range(20)]
    ingested_ids = []
    try:
        for image_path, photo_id in placed[:3]:
            assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
            ingested_ids.append(photo_id)

        # 17 dosya HALA inbox'ta - "yukleme suruyor" durumu.
        remaining = {p.name for p in ingestion_service.list_ready_files()}
        assert len(remaining) >= 17, f"17 dosya inbox'ta beklenirdi, {len(remaining)} var"

        # ASIL IDDIA: kuyrukta, worker'in HEMEN alabilecegi is VAR.
        #
        # Neden dogrudan claim_next cagirip "None gelmedi" demiyoruz: bu
        # makinede GERCEK worker surecleri kosuyor ve testin isini kendi
        # alabiliyor - o zaman claim_next bize None doner ve test, dogru
        # calisan bir sistemde YANLIS YERE kalir. Bunun yerine kuyrugun
        # gozlemlenebilir durumu sorgulanir: bu, claim'i kimin yaptigindan
        # BAGIMSIZ olarak "is hazir hale geldi mi" sorusunu yanitlar.
        db = SessionLocal()
        try:
            available = db.execute(
                text(
                    """
                    SELECT count(*) FROM jobs
                    WHERE type = :t
                      AND run_after <= now()
                      AND (payload->>'photo_id')::uuid = ANY(:ids)
                    """
                ),
                {"t": JOB_TYPE_FACE_PIPELINE, "ids": ingested_ids},
            ).scalar_one()
        finally:
            db.close()

        assert available == 3, (
            f"Ingest edilen 3 fotograf icin 3 alinabilir is beklenirdi, {available} var. "
            "Is kayitlari, 20 fotografin TAMAMI beklenmeden olusmali - aksi halde "
            "hat producer/consumer degil, toplu-bekleme mantigiyla calisiyor demektir."
        )
    finally:
        _purge([pid for _, pid in placed], test_user_id)
        _drain_queue(test_user_id)


# --- TEST 8 -----------------------------------------------------------


def test_worker_consumes_db_queue_not_inbox(test_user_id):
    """TEST 8 - inbox'a dosya koymak TEK BASINA is uretmez.

    Worker'in tukettigi kaynak DB kuyrugudur; inbox yalnizca ingestion'in
    girdisidir. Bu ayrim bozulursa (or. worker klasoru taramaya baslarsa)
    ingest edilmemis bir dosya icin de is alinabilirdi.
    """
    _drain_queue(test_user_id)
    image_path, photo_id = _place_in_inbox(test_user_id)
    try:
        assert image_path.exists(), "dosya inbox'ta hazir bekliyor"

        claimed = jr.claim_next("test-worker-inbox-blind", [JOB_TYPE_FACE_PIPELINE])
        assert claimed is None, (
            "Ingest EDILMEMIS bir inbox dosyasi icin is alinabildi - worker "
            "DB kuyrugu disinda bir kaynaktan is uretiyor olabilir"
        )

        # Ingestion kosunca is olusur ve ANCAK O ZAMAN alinabilir.
        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
        claimed = jr.claim_next("test-worker-inbox-blind", [JOB_TYPE_FACE_PIPELINE])
        assert claimed is not None, "ingestion sonrasi is alinabilmeliydi"
        jr.complete(claimed.id, "test-worker-inbox-blind")
    finally:
        _purge([photo_id], test_user_id)
        _drain_queue(test_user_id)


# --- TEST 6 -----------------------------------------------------------


def test_identical_content_ingested_twice_creates_one_photo(test_user_id):
    """TEST 6 - ayni icerik, IKI ayri inbox dosyasi (iki farkli photo_id).

    Duplicate on-elemesi upload asamasinda yapilir ama tam es zamanli iki
    yukleme ikisini de gecebilir; bu test o durumu birebir kurar (ikisi de
    inbox'a dusmus). Sonuc: TEK Photo satiri, TEK is cifti.
    """
    _drain_queue(test_user_id)
    shared = _random_jpeg_bytes()
    path_a, id_a = _place_in_inbox(test_user_id, data=shared)
    path_b, id_b = _place_in_inbox(test_user_id, data=shared)
    try:
        first = ingestion_service.ingest_file(path_a)
        second = ingestion_service.ingest_file(path_b)

        assert first.outcome == IngestOutcome.CREATED
        assert second.outcome == IngestOutcome.DUPLICATE, (
            f"ikinci dosya duplicate sayilmaliydi, {second.outcome} dondu"
        )
        assert second.photo_id == id_a, "duplicate, MEVCUT kaydi isaret etmeli"

        db = SessionLocal()
        try:
            content_hash = ingestion_service.hash_file(photo_service.STORED_DIR / f"{id_a}.jpg")
            rows = db.execute(
                text("SELECT count(*) FROM photos WHERE content_hash = :h"), {"h": content_hash}
            ).scalar_one()
            assert rows == 1, f"ayni icerik icin {rows} Photo satiri var, 1 olmaliydi"

            jobs_b = db.execute(
                text("SELECT count(*) FROM jobs WHERE payload->>'photo_id' = :p"),
                {"p": str(id_b)},
            ).scalar_one()
            assert jobs_b == 0, "duplicate icin is kaydi OLUSTURULMAMALIYDI"
        finally:
            db.close()

        assert not path_b.exists(), "duplicate inbox dosyasi temizlenmeliydi"
        assert not photo_service._meta_path_for(path_b).exists()
    finally:
        _purge([id_a, id_b], test_user_id)
        _drain_queue(test_user_id)


def test_db_unique_constraint_is_the_final_guarantee():
    """Spec 6 - 'once SELECT sonra INSERT' tek basina yeterli SAYILMAZ.

    Uygulama katmanini tamamen atlayip dogrudan iki satir INSERT etmeyi
    dener; DB kisiti bunu REDDETMELI.
    """
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex[:0]  # 32 hex; kolon 64'e kadar
    a, b = uuid.uuid4(), uuid.uuid4()
    db = SessionLocal()
    try:
        ins = text(
            "INSERT INTO photos (id, filename, storage_path, status, content_hash) "
            "VALUES (:i, :f, :s, 'processing', :h)"
        )
        db.execute(ins, {"i": str(a), "f": "u1.jpg", "s": "x/u1.jpg", "h": content_hash})
        db.commit()

        raised = False
        try:
            db.execute(ins, {"i": str(b), "f": "u2.jpg", "s": "x/u2.jpg", "h": content_hash})
            db.commit()
        except Exception:
            raised = True
            db.rollback()
        assert raised, (
            "photos.content_hash uzerinde UNIQUE kisit YOK - tekillik yalnizca "
            "uygulama katmaninda kaliyor (migration b4d7e2a9c153 uygulanmis mi?)"
        )
    finally:
        db.execute(text("DELETE FROM photos WHERE id IN (:a, :b)"), {"a": str(a), "b": str(b)})
        db.commit()
        db.close()


# --- TEST 4 / crash kurtarma -------------------------------------------


def test_orphan_meta_after_commit_is_cleaned(test_user_id):
    """TEST 4a - commit BASARILI, meta silinmeden crash.

    Restart'ta: kayit var, dosya stored'da. Yapilacak tek sey oksuz meta'yi
    temizlemek - dosya GERI ALINMAMALI.
    """
    image_path, photo_id = _place_in_inbox(test_user_id)
    try:
        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
        # Crash simulasyonu: meta'yi geri yaz (silinmemis gibi)
        photo_service._write_inbox_meta(image_path, {"photo_id": str(photo_id)})

        healed = ingestion_service.recover_orphan_meta()
        assert healed >= 1
        assert not photo_service._meta_path_for(image_path).exists(), "oksuz meta silinmeliydi"
        assert (photo_service.STORED_DIR / f"{photo_id}.jpg").exists(), (
            "commit basariliydi - dosya stored/'da KALMALI, geri alinmamali"
        )
    finally:
        _purge([photo_id], test_user_id)
        _drain_queue(test_user_id)


def test_orphan_meta_before_commit_returns_file_to_inbox(test_user_id):
    """TEST 4b - dosya stored/'a tasindi ama commit'ten ONCE crash.

    Restart'ta: kayit YOK, dosya stored'da, meta inbox'ta. Dosya inbox'a
    GERI ALINMALI ki bastan islensin - aksi halde hicbir kaydin isaret
    etmedigi oksuz bir dosya olarak sonsuza kadar kalirdi.
    """
    image_path, photo_id = _place_in_inbox(test_user_id)
    stored_path = photo_service.STORED_DIR / f"{photo_id}.jpg"
    try:
        # Crash'i elle kur: dosyayi tasi, kaydi OLUSTURMA, meta'yi birak.
        os.replace(image_path, stored_path)
        assert not image_path.exists()

        healed = ingestion_service.recover_orphan_meta()
        assert healed >= 1
        assert image_path.exists(), "commit oncesi crash'te dosya inbox'a GERI ALINMALIYDI"
        assert not stored_path.exists()

        # Ve artik normal sekilde islenebilmeli.
        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
        db = SessionLocal()
        try:
            assert db.query(Photo).filter(Photo.id == photo_id).first() is not None
        finally:
            db.close()
    finally:
        _purge([photo_id], test_user_id)
        _drain_queue(test_user_id)


def test_failed_ingest_does_not_leak_content_lock(test_user_id, monkeypatch):
    """REGRESYON - ingest hatasi advisory kilidi SIZDIRMAMALI.

    Ilk uygulamada kilit session-scoped'di ve hata yolundaki `db.rollback()`
    baglantiyi havuza iade ettigi icin, ardindan gelen unlock BASKA bir
    baglantida calisiyordu. Kilit havuzdaki eski baglantida asili kaliyor ve
    o content_hash BIR DAHA ASLA ingest edilemiyordu - uretimde sessiz,
    kalici bir veri kaybi olurdu. Cozum: transaction-scoped kilit
    (pg_try_advisory_xact_lock), bkz. app/db/locks.py.
    """
    _drain_queue(test_user_id)
    image_path, photo_id = _place_in_inbox(test_user_id)
    try:
        def boom(*a, **kw):
            raise RuntimeError("simule edilmis gecici DB hatasi")

        monkeypatch.setattr(ingestion_service.jobs_repository, "enqueue", boom)
        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.FAILED
        monkeypatch.undo()

        # Kilit birakilmis olmali: ayni icerik sorunsuz ingest edilebilmeli.
        result = ingestion_service.ingest_file(image_path)
        assert result.outcome == IngestOutcome.CREATED, (
            f"hata sonrasi kilit SIZDI - ikinci deneme {result.outcome!r} "
            f"dondu (detay={result.detail!r})"
        )

        # Ve DB tarafinda gercekten asili kilit kalmamis olmali.
        db = SessionLocal()
        try:
            leaked = db.execute(
                text(
                    "SELECT count(*) FROM pg_locks WHERE locktype='advisory' "
                    "AND classid = :c"
                ),
                {"c": ingestion_service.PHOTOAI_LOCK_CLASS_INGEST},
            ).scalar_one()
            assert leaked == 0, f"{leaked} adet ingest advisory kilidi asili kalmis"
        finally:
            db.close()
    finally:
        _purge([photo_id], test_user_id)
        _drain_queue(test_user_id)


# --- yarim dosya korumasi ----------------------------------------------


def test_part_files_are_never_listed_as_ready(test_user_id):
    """Yaziliyor olan (.part) dosya ISLENMEYE HAZIR sayilmamali - yarim
    dosyanin ingest edilmesi yapisal olarak imkansiz olmali."""
    part = photo_service.INBOX_DIR / f"{uuid.uuid4()}.jpg{photo_service.PART_SUFFIX}"
    part.write_bytes(b"\xff\xd8yarim veri")
    try:
        names = {p.name for p in ingestion_service.list_ready_files()}
        assert part.name not in names, ".part dosyasi hazir listesine GIRMEMELI"
    finally:
        part.unlink(missing_ok=True)


def test_stale_part_is_swept_to_failed(test_user_id):
    """Istek ortasinda backend kapanirsa geride kalan .part, belirli bir
    sure sonra failed/'a alinmali - inbox'ta sonsuza kadar birikmemeli."""
    part = photo_service.INBOX_DIR / f"{uuid.uuid4()}.jpg{photo_service.PART_SUFFIX}"
    part.write_bytes(b"olu yukleme")
    moved = photo_service.FAILED_DIR / part.name
    try:
        # Yeni dosya supurulmemeli (su an yazilan bir yukleme olabilir).
        assert ingestion_service.sweep_stale_parts(3600) == 0
        assert part.exists()

        # Dosyayi ACIKCA eskitiyoruz. Onceki hali `sweep_stale_parts(0)`
        # cagiriyordu ve ARALIKLI olarak kaliyordu: Windows'ta dosya mtime'i
        # time.time()'in milisaniyeler ONUNDE olabiliyor, boylece
        # (now - mtime) NEGATIF cikip "0'dan kucuk" testine takiliyor ve
        # dosya yasli SAYILMIYORDU. Esikle oynamak yerine dosyanin yasini
        # kesinlestirmek, testin olcmek istedigi seye de daha sadik.
        old = time.time() - 7200
        os.utime(part, (old, old))

        assert ingestion_service.sweep_stale_parts(3600) >= 1
        assert not part.exists()
        assert moved.exists(), "olu .part failed/ altina alinmaliydi"
    finally:
        part.unlink(missing_ok=True)
        moved.unlink(missing_ok=True)
