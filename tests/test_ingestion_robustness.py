"""Ingestion dayaniklilik testleri (production review'da bulunan sorunlar).

Buradaki her test, gozden gecirmede tespit edilmis SOMUT bir arizayi
temsil eder - hepsi once KIRMIZI yazildi, sonra duzeltildi.
"""

import io
import os
import uuid
from pathlib import Path

from PIL import Image
from sqlalchemy import text

from app.db.models import Photo
from app.db.session import SessionLocal
from app.services import ingestion_service, photo_service
from app.services.ingestion_service import IngestOutcome


def _jpeg() -> bytes:
    img = Image.new("RGB", (8, 8))
    img.putdata([tuple(os.urandom(3)) for _ in range(64)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _place(user_id, data: bytes | None = None) -> tuple[Path, uuid.UUID]:
    photo_id = uuid.uuid4()
    data = data if data is not None else _jpeg()
    p = photo_service.INBOX_DIR / f"{photo_id}.jpg"
    p.write_bytes(data)
    photo_service._write_inbox_meta(
        p,
        {
            "photo_id": str(photo_id),
            "original_filename": f"robustness-{photo_id}.jpg",
            "content_hash": ingestion_service.hash_file(p),
            "size_bytes": len(data),
            "uploaded_by_user_id": str(user_id),
            "received_at": "2026-09-04T00:00:00",
        },
    )
    return p, photo_id


def _purge(ids, user_id):
    db = SessionLocal()
    try:
        for pid in ids:
            db.execute(text("DELETE FROM jobs WHERE payload->>'photo_id' = :p"), {"p": str(pid)})
            db.execute(text("DELETE FROM activity_log WHERE target_id=:p"), {"p": str(pid)})
            db.execute(text("DELETE FROM photo_exif WHERE photo_id=:p"), {"p": str(pid)})
            db.execute(text("DELETE FROM photos WHERE id=:p"), {"p": str(pid)})
        db.execute(text("DELETE FROM jobs WHERE user_id = :u"), {"u": str(user_id)})
        db.commit()
    finally:
        db.close()


# --- ARIZA A: bozuk tek meta, TUM uzlastirmayi kilitliyor --------------


def test_corrupt_meta_does_not_block_reconciliation(test_user_id):
    """Bir bozuk meta yan-dosyasi, DIGER dosyalarin uzlastirilmasini
    ENGELLEMEMELI.

    ARIZA: recover_orphan_meta dongusunde `uuid.UUID(photo_id)` per-oge
    korumasizdi. Bozuk bir meta (elle olusturulmus, disk bozulmasi, eski
    surumden kalma) her turda ayni istisnayi firlatiyor; _reconcile bunu
    yakalayip logluyor ama dongu O NOKTADA kesiliyordu. Sirali listede
    ondan SONRA gelen oksuz metalar BIR DAHA ASLA uzlastirilmiyordu -
    yani crash kurtarma kalici olarak duruyordu.
    """
    # Sirada ONCE gelsin diye adi '0' ile baslayan bozuk bir meta.
    bad_image = photo_service.INBOX_DIR / "0000-bozuk.jpg"
    bad_meta = photo_service._meta_path_for(bad_image)
    bad_meta.write_text('{"photo_id": "BU-BIR-UUID-DEGIL"}', encoding="utf-8")

    # Ardindan GERCEK bir kurtarma vakasi: dosya stored/'da, kayit yok,
    # meta inbox'ta (commit oncesi crash).
    image_path, photo_id = _place(test_user_id)
    stored_path = photo_service.STORED_DIR / f"{photo_id}.jpg"
    os.replace(image_path, stored_path)

    try:
        ingestion_service.recover_orphan_meta()

        assert image_path.exists(), (
            "Bozuk bir meta yuzunden GERCEK kurtarma vakasi islenmedi - "
            "tek bozuk dosya crash kurtarmayi kalici olarak durduruyor"
        )
        assert not bad_meta.exists(), "bozuk meta temizlenmeliydi"
    finally:
        bad_meta.unlink(missing_ok=True)
        stored_path.unlink(missing_ok=True)
        image_path.unlink(missing_ok=True)
        photo_service._meta_path_for(image_path).unlink(missing_ok=True)
        _purge([photo_id], test_user_id)


# --- ARIZA B: ucustaki yukleme 'oksuz meta' sayiliyor -------------------


def test_inflight_upload_is_not_counted_as_orphan_meta(test_user_id):
    """Devam eden bir yukleme (.part + .meta) OKSUZ META sayilmamali.

    ARIZA: inbox_stats(), oksuz meta'yi `metas - ready` ile hesapliyordu.
    receive_upload meta'yi nihai yeniden adlandirmadan ONCE yazdigi icin,
    SU AN yuklenen her dosya bir 'oksuz meta' olarak gorunuyordu. 12 es
    zamanli yuklemede operator `orphan_meta: 12` goruyordu - ki bu metrigin
    TEK anlami 'crash yasandi'. Yanlis alarm.
    """
    photo_id = uuid.uuid4()
    part = photo_service.INBOX_DIR / f"{photo_id}.jpg{photo_service.PART_SUFFIX}"
    final = photo_service.INBOX_DIR / f"{photo_id}.jpg"
    part.write_bytes(b"yariya kadar yazildi")
    photo_service._write_inbox_meta(final, {"photo_id": str(photo_id)})
    try:
        stats = ingestion_service.inbox_stats()
        assert stats["in_progress"] == 1, "devam eden yukleme sayilmali"
        assert stats["orphan_meta"] == 0, (
            f"devam eden yukleme OKSUZ META sayildi (orphan_meta={stats['orphan_meta']}) - "
            "bu metrigin tek anlami 'crash yasandi' olmali"
        )
    finally:
        part.unlink(missing_ok=True)
        photo_service._meta_path_for(final).unlink(missing_ok=True)


def test_real_orphan_meta_is_still_reported(test_user_id):
    """Yukaridaki duzeltme GERCEK oksuz metayi gizlememeli."""
    photo_id = uuid.uuid4()
    final = photo_service.INBOX_DIR / f"{photo_id}.jpg"
    photo_service._write_inbox_meta(final, {"photo_id": str(photo_id)})
    try:
        assert ingestion_service.inbox_stats()["orphan_meta"] == 1
    finally:
        photo_service._meta_path_for(final).unlink(missing_ok=True)


# --- ARIZA C: yarisi kaybeden dongu sahte 'failed' uretiyor -------------


def test_losing_a_race_is_skipped_not_failed(test_user_id):
    """Dosyayi baska bir ingestion dongusu almissa sonuc SKIPPED olmali,
    FAILED degil.

    ARIZA: coklu dongude (uvicorn --workers, ya da --reload'un gecici cift
    sureci) kaybeden dongu, meta'si silinmis ve dosyasi tasinmis bir yolu
    gorup `_move_to_failed` cagiriyordu. Sonuc: var olmayan dosya uzerinde
    exception traceback'i + 'failed' metriginin sahte artmasi. Operator
    icin gercek arizadan ayirt edilemez bir gurultu.
    """
    ghost = photo_service.INBOX_DIR / f"{uuid.uuid4()}.jpg"  # hic olusturulmadi
    result = ingestion_service.ingest_file(ghost)
    assert result.outcome == IngestOutcome.SKIPPED, (
        f"kaybedilen yaris SKIPPED olmaliydi, {result.outcome} dondu - "
        "sahte 'failed' metrigi uretiyor"
    )
    assert not (photo_service.FAILED_DIR / ghost.name).exists()


def test_externally_dropped_file_without_meta_is_imported(test_user_id):
    """META ZORUNLU DEGIL - inbox'a DOGRUDAN birakilmis dosya ice aktarilmali.

    Onceki davranis bu dosyayi failed/'a aliyordu ("yukleyiciyi bilemeyiz").
    On-premise'te inbox'a klasor kopyalamak mesru bir toplu ice aktarim yolu;
    o yoldan gelen SAGLAM fotograflarin sessizce failed/'a dusmesi gercek bir
    veri kaybi riskiydi. Artik meta sentezleniyor ve kayit, sahte bir insan
    sahipligi yerine SISTEM SERVIS HESABINA yaziliyor.
    """
    from app.core.settings import settings

    data = _jpeg()
    dropped = photo_service.INBOX_DIR / f"{uuid.uuid4()}.jpg"
    dropped.write_bytes(data)
    created_id = None
    try:
        result = ingestion_service.ingest_file(dropped)
        assert result.outcome == IngestOutcome.CREATED, (
            f"dogrudan birakilan dosya ice aktarilmaliydi, {result.outcome} dondu"
        )
        created_id = result.photo_id
        assert not (photo_service.FAILED_DIR / dropped.name).exists()

        db = SessionLocal()
        try:
            row = db.query(Photo).filter(Photo.id == created_id).first()
            assert row is not None
            assert str(row.uploaded_by_user_id) == settings.SYSTEM_USER_ID, (
                "sahiplik SISTEM servis hesabina yazilmali - sahte insan sahipligi degil"
            )
            assert row.filename == dropped.name
            jobs = db.execute(
                text("SELECT count(*) FROM jobs WHERE payload->>'photo_id' = :p"),
                {"p": str(created_id)},
            ).scalar_one()
            assert jobs == 2, "face_pipeline + vlm_analysis olusmali"
        finally:
            db.close()

        stored = photo_service.STORED_DIR / dropped.name
        assert stored.exists() and stored.read_bytes() == data
    finally:
        dropped.unlink(missing_ok=True)
        (photo_service.STORED_DIR / dropped.name).unlink(missing_ok=True)
        photo_service._meta_path_for(dropped).unlink(missing_ok=True)
        if created_id:
            _purge([created_id], test_user_id)


def test_synthesized_meta_is_written_before_move(test_user_id):
    """Sentezlenen meta DISKE de yazilmali - crash kurtarmanin dayandigi
    "ingestion bitmedi" isareti odur. Yazilmazsa, commit oncesi bir crash
    geriye hicbir iz birakmaz ve dosya oksuz kalirdi."""
    dropped = photo_service.INBOX_DIR / f"{uuid.uuid4()}.jpg"
    dropped.write_bytes(_jpeg())
    try:
        meta = ingestion_service._synthesize_meta(dropped)
        assert photo_service._meta_path_for(dropped).exists(), (
            "sentezlenen meta diske yazilmali"
        )
        assert meta["source"] == "external_drop", "koken acikca isaretlenmeli"
        # Diskteki meta okunabilir ve ayni olmali
        assert photo_service.read_inbox_meta(dropped)["photo_id"] == meta["photo_id"]
    finally:
        dropped.unlink(missing_ok=True)
        photo_service._meta_path_for(dropped).unlink(missing_ok=True)


# --- dosya kaybi: hicbir yolda veri yok olmamali ------------------------


def test_no_data_loss_across_repeated_ingest_of_same_file(test_user_id):
    """Ayni dosya defalarca islenebilmeli ve icerik HICBIR asamada
    kaybolmamali (idempotans)."""
    data = _jpeg()
    image_path, photo_id = _place(test_user_id, data=data)
    stored = photo_service.STORED_DIR / f"{photo_id}.jpg"
    try:
        assert ingestion_service.ingest_file(image_path).outcome == IngestOutcome.CREATED
        assert stored.read_bytes() == data, "icerik degismis olmamali"

        # 2. ve 3. kez: dosya artik inbox'ta yok -> SKIPPED, veri duruyor.
        for _ in range(2):
            ingestion_service.ingest_file(image_path)
            assert stored.exists() and stored.read_bytes() == data

        db = SessionLocal()
        try:
            n = db.execute(
                text("SELECT count(*) FROM photos WHERE id = :p"), {"p": str(photo_id)}
            ).scalar_one()
            assert n == 1, "tekrar isleme IKINCI bir kayit uretmemeli"
        finally:
            db.close()
    finally:
        stored.unlink(missing_ok=True)
        _purge([photo_id], test_user_id)


# --- GERCEK eszamanli duplicate yarisi ---------------------------------


def test_concurrent_ingest_of_identical_content_creates_one_photo(test_user_id):
    """Ayni icerik, IKI dosya, IKI THREAD, AYNI ANDA.

    Onceki duplicate testi ardisikti - o, kilidin/UNIQUE'in gercek yaris
    altindaki davranisini OLCMUYORDU. Burada iki thread bariyerle
    hizalanip ayni anda baslar.

    Beklenen: TEK Photo satiri, TEK is cifti; iki thread'den biri CREATED,
    digeri DUPLICATE ya da SKIPPED (kilidi alamamis) doner - ama HICBIR
    kosulda iki kayit olusmaz.
    """
    import threading

    shared = _jpeg()
    path_a, id_a = _place(test_user_id, data=shared)
    path_b, id_b = _place(test_user_id, data=shared)
    barrier = threading.Barrier(2)
    results = {}

    def worker(name, path):
        barrier.wait()
        results[name] = ingestion_service.ingest_file(path)

    try:
        ta = threading.Thread(target=worker, args=("a", path_a))
        tb = threading.Thread(target=worker, args=("b", path_b))
        ta.start(); tb.start(); ta.join(20); tb.join(20)

        outcomes = sorted(r.outcome for r in results.values())
        content_hash = ingestion_service.hash_file(
            photo_service.STORED_DIR / f"{id_a}.jpg"
            if (photo_service.STORED_DIR / f"{id_a}.jpg").exists()
            else photo_service.STORED_DIR / f"{id_b}.jpg"
        )

        db = SessionLocal()
        try:
            rows = db.execute(
                text("SELECT count(*) FROM photos WHERE content_hash = :h"),
                {"h": content_hash},
            ).scalar_one()
            jobs = db.execute(
                text(
                    "SELECT count(*) FROM jobs WHERE (payload->>'photo_id')::uuid "
                    "IN (:a, :b)"
                ).bindparams(a=str(id_a), b=str(id_b))
            ).scalar_one()
        finally:
            db.close()

        assert rows == 1, (
            f"eszamanli yarista {rows} Photo satiri olustu, 1 olmaliydi "
            f"(sonuclar: {outcomes})"
        )
        assert jobs == 2, f"is kaydi 2 olmaliydi (face+vlm), {jobs} var"
        assert IngestOutcome.CREATED in outcomes, f"biri CREATED olmaliydi: {outcomes}"
    finally:
        for pid in (id_a, id_b):
            (photo_service.STORED_DIR / f"{pid}.jpg").unlink(missing_ok=True)
            (photo_service.INBOX_DIR / f"{pid}.jpg").unlink(missing_ok=True)
            photo_service._meta_path_for(photo_service.INBOX_DIR / f"{pid}.jpg").unlink(missing_ok=True)
        _purge([id_a, id_b], test_user_id)


def test_duplicate_with_corrupt_stored_file_is_not_deleted(test_user_id):
    """Duplicate silme GERI ALINAMAZ - hedef bozuksa inbox kopyasi SILINMEZ.

    Senaryo: DB'de content_hash eslesen bir kayit var ama diskteki dosyasi
    bozulmus/kesilmis. Inbox kopyasi o an tek SAGLAM kopya olabilir; korumasiz
    bir `unlink` burada sessiz veri kaybi olurdu. Beklenen: failed/'a alinir.
    """
    data = _jpeg()
    first, id_first = _place(test_user_id, data=data)
    stored = photo_service.STORED_DIR / f"{id_first}.jpg"
    second, id_second = _place(test_user_id, data=data)
    moved = photo_service.FAILED_DIR / second.name
    try:
        assert ingestion_service.ingest_file(first).outcome == IngestOutcome.CREATED
        # stored dosyasini BOZ (kesilmis yazim taklidi)
        stored.write_bytes(data[: len(data) // 2])

        result = ingestion_service.ingest_file(second)
        assert result.outcome == IngestOutcome.FAILED, (
            f"bozuk hedefe ragmen {result.outcome} dondu - inbox kopyasi silinmis olabilir"
        )
        assert not second.exists(), "dosya failed/'a tasinmis olmali"
        assert moved.exists(), "inbox kopyasi KORUNMALI (failed/ altinda)"
        assert moved.read_bytes() == data, "korunan kopya bozulmamis olmali"
    finally:
        stored.unlink(missing_ok=True)
        moved.unlink(missing_ok=True)
        photo_service._meta_path_for(moved).unlink(missing_ok=True)
        _purge([id_first, id_second], test_user_id)
