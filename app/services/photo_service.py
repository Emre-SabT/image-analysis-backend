import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile
from qdrant_client.models import PointStruct
from sqlalchemy.orm import Session
from PIL import Image, ImageOps
import pillow_heif

from app.db import identity_locks, locks, qdrant
from app.db.models import ClusterConstraint, Face, Photo, PhotoAnalysis
from app.ai.dispatcher import analyze_photo
from app.services import face_service, person_service
from app.core.settings import settings

logger = logging.getLogger("photoai.photo_service")

pillow_heif.register_heif_opener()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CONVERTED_DIR = UPLOAD_DIR / "converted"
CONVERTED_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


class LockConflict(Exception):
    """PR-2: bu (is tipi, photo_id) kaynagini O AN baska bir worker
    isliyor (pg_try_advisory_lock False dondu). GERCEK bir hata DEGIL -
    handler hic calismadi. worker/main.py bunu jobs.fail() yerine
    jobs.requeue_lock_conflict() ile cezasiz kuyruga koyar (bkz.
    _process()'teki ayri except bloğu)."""


def save_upload(db: Session, file: UploadFile, uploaded_by_user_id: uuid.UUID | None = None) -> tuple[Photo, bool]:
    """Dosyanin SHA-256 hash'ini hesaplar, diske YAZMADAN ONCE ayni icerigin
    (bayt bayt) daha once yuklenip yuklenmedigini kontrol eder.

    Dosya tekillestirme: aynı fotoğraf birden fazla yüklenirse yüz/analiz
    kayıtları da tekrarlanıyordu (dosya adı degil, ICERIK karsilastirilir -
    ayni foto farkli isimle tekrar yuklense bile yakalanir). Eslesme varsa
    hicbir yeni dosya/DB kaydi olusturulmadan MEVCUT Photo dondurulur.

    Yaris durumu notu: content_hash kolonu DB'de UNIQUE degil (bkz. models.py -
    bu ozellik eklenmeden once yuklenmis, halihazirda yinelenen fotograflar
    yuzunden), bu yuzden tam es zamanli iki ayni-dosya yuklemesi teorik olarak
    ikisi de gecebilir. Bu uygulama tek kullanicili/yerel oldugundan ve
    yuklemeler frontend'de sirali (await) yapildigindan pratikte bir risk
    tasimiyor.

    Donus: (photo, is_duplicate)
    """
    content = file.file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    existing = db.query(Photo).filter(Photo.content_hash == content_hash).first()
    if existing:
        return existing, True

    ext = Path(file.filename).suffix.lower()
    photo_id = uuid.uuid4()
    stored_name = f"{photo_id}{ext}"
    storage_path = UPLOAD_DIR / stored_name

    with open(storage_path, "wb") as out:
        out.write(content)

    photo = Photo(
        id=photo_id,
        filename=file.filename,
        storage_path=str(storage_path),
        status="processing",
        content_hash=content_hash,
        uploaded_by_user_id=uploaded_by_user_id,
    )
    db.add(photo)
    # COMMIT DEGIL FLUSH: commit sorumlulugu CAGIRANA (router) ait.
    # POST /photos, foto satirini ve IKI job satirini TEK atomik commit'te
    # yazmali - job insert patlarsa foto insert de geri alinmali.
    db.flush()
    return photo, False




def run_face_pipeline(db: Session, photo: Photo) -> list:
    """A5-A8: yuz tespiti + embedding + kimlik atama.

    VLM'den BAGIMSIZ ve ucuz (~0.13 sn/fotograf) oldugu icin yukleme isteginin
    icinde calisir - boylece 'Kisiler' ekrani aninda dolar, kullanici VLM'i
    beklemeden isimlendirmeye baslayabilir (Bolum 10.1 hata yari-gecirgenligi).
    """
    try:
        return face_service.detect_and_embed(db, photo)
    except Exception as e:
        print(f"[YUZ TESPITI HATASI] {photo.filename}: {type(e).__name__}: {e}")
        return []


async def run_vlm_analysis(db: Session, photo: Photo) -> Photo:
    """VLM'i cagirir, sonucu photo_analysis'e yazar. Yuz hattini calistirmaz.

    IDEMPOTENT: kuyruk at-least-once oldugu icin (reaper ayni isi yeniden
    kuyruga koyabilir) bu fonksiyon ayni photo_id icin birden fazla kez
    calisabilmeli. Mevcut analiz satiri varsa once SILINIR, sonra yenisi
    yazilir - photo_analysis.photo_id birincil anahtar oldugundan aksi
    halde ikinci calisma PK ihlaliyle patlardi.
    """
    try:
        result = await analyze_photo(photo.storage_path)

        # Idempotency: varsa eski satiri temizle (upsert davranisi).
        db.query(PhotoAnalysis).filter(
            PhotoAnalysis.photo_id == photo.id
        ).delete(synchronize_session=False)

        analysis = PhotoAnalysis(
            photo_id=photo.id,
            description=result.description,
            environment_type=result.environment_type,
            people_count=result.people_count,
            possible_event=result.possible_event,
            primary_object=result.primary_object,
            secondary_objects=result.secondary_objects,
            environment=result.environment,
            attributes=result.attributes,
            action=result.action,
            mood=result.mood,
            use_case=result.use_case,
            context=result.context,
            style=result.style,
            audience=result.audience,
            public_figures=[pf.model_dump() for pf in result.public_figures],
            all_tags=result.all_tags,
            model_name=settings.VLM_MODEL,
            analyzed_at=datetime.utcnow(),
        )
        db.add(analysis)
        photo.status = "analyzed"
    except Exception as e:
        print(f"[ANALIZ HATASI] {photo.filename}: {type(e).__name__}: {e}")
        photo.status = "failed"

    db.commit()
    db.refresh(photo)
    return photo


def get_photo_with_analysis(db: Session, photo_id: uuid.UUID):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        return None, None
    analysis = db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == photo_id).first()
    return photo, analysis


def list_photos(db: Session):
    photos = db.query(Photo).order_by(Photo.created_at.desc()).all()
    out = []
    for p in photos:
        a = db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == p.id).first()
        out.append((p, a))
    return out


def delete_photo(db: Session, photo_id: uuid.UUID) -> dict:
    """Bir fotografi ve ondan turemis TUM veriyi siler.

    Silinenler: fotograf dosyasi (+ HEIC onbellegi), photo_analysis kaydi,
    fotograftaki tum yuz kayitlari (kirpim dosyalari + Qdrant vektorleri +
    FK bagimliliklari) ve photos satiri.

    Onemli: Silinen yuzlerin ait oldugu kisi/klasorlerin merkezleri KALAN
    uyelerle yeniden hesaplanir; son uyesi de gitmisse o kimlik tamamen
    silinir. Aksi halde artik var olmayan bir yuzun katkisi kisinin
    merkezinde kalmaya devam ederdi.
    """
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if photo is None:
        raise ValueError("Fotograf bulunamadi")

    faces = db.query(Face).filter(Face.photo_id == photo_id).all()
    face_ids = [f.id for f in faces]

    # Etkilenen kimlikler, yuzler silinmeden ONCE toplanmali.
    affected_persons = {f.person_id for f in faces if f.person_id}
    affected_clusters = {f.cluster_id for f in faces if f.cluster_id}

    # PR-D, KATMAN 1: TUM etkilenen kimlikler TEK cagrida, ic siralamayla
    # kilitlenir (bkz. identity_locks.py) - bir worker'in eszamanli artimli
    # guncellemesine karsi, asagidaki recompute-or-delete yazimlarini korur.
    identity_locks.lock_identities(
        db,
        [("person", pid) for pid in affected_persons] + [("cluster", cid) for cid in affected_clusters],
    )

    if face_ids:
        db.query(ClusterConstraint).filter(
            (ClusterConstraint.face_id_a.in_(face_ids))
            | (ClusterConstraint.face_id_b.in_(face_ids))
        ).delete(synchronize_session=False)

        for face in faces:
            crop = Path(face.crop_path)
            if crop.exists():
                try:
                    crop.unlink()
                except OSError:
                    pass  # dosya silinemezse DB temizligi yine de surmeli

        qdrant.client.delete(
            collection_name=qdrant.FACES_COLLECTION,
            points_selector=[str(fid) for fid in face_ids],
        )
        db.query(Face).filter(Face.id.in_(face_ids)).delete(synchronize_session=False)
        db.flush()  # merkez hesabi guncel durumu gormeli

    # YIKICI dal (bossa kalanlar) Qdrant'tan HEMEN silinir (yukaridaki
    # helper'larin kendi ici); YARATICI dal (uye kalanlar) SADECE PG'yi
    # gunceller, centroid asagida commit SONRASI GUNCEL uyelikle yazilir
    # (Aşama 1 duzeltmesi - bkz. person_service._recompute_or_delete_person
    # docstring'i).
    persons_needing_recompute: set[uuid.UUID] = set()
    clusters_needing_recompute: set[uuid.UUID] = set()
    for person_id in affected_persons:
        if person_service._recompute_or_delete_person(db, person_id):
            persons_needing_recompute.add(person_id)
    for cluster_id in affected_clusters:
        if person_service._recompute_or_delete_cluster(db, cluster_id):
            clusters_needing_recompute.add(cluster_id)

    db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == photo_id).delete(
        synchronize_session=False
    )

    for path in (Path(photo.storage_path), CONVERTED_DIR / f"{photo.id}.jpg"):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    db.query(Photo).filter(Photo.id == photo_id).delete(synchronize_session=False)
    db.commit()

    # PR-D, KATMAN 2: uyesi kalan (silinmeyen) her kisi/kume icin KENDI
    # kilidini alir, uyeligi TAZE sorgular (yukaridaki member_ids DEGIL - bu
    # fonksiyon artik face_ids parametre almiyor), PG centroid'i yazar.
    # Donen dict'ler commit SONRASI (Faz 3) Qdrant dual-write icin.
    pending_identity_ops = []
    for person_id in persons_needing_recompute:
        op = person_service._upsert_identity_centroid(db, "person", person_id)
        if op is not None:
            pending_identity_ops.append(op)
    for cluster_id in clusters_needing_recompute:
        op = person_service._upsert_identity_centroid(db, "cluster", cluster_id)
        if op is not None:
            pending_identity_ops.append(op)

    for op in pending_identity_ops:
        qdrant.client.upsert(
            collection_name=qdrant.IDENTITY_POOL_COLLECTION,
            points=[PointStruct(id=op["identity_id"], vector=op["centroid"], payload=op["payload"])],
        )

    return {"deleted_photo_id": str(photo_id), "deleted_faces": len(face_ids)}


def get_servable_file(photo: Photo) -> tuple[Path, str | None]:
    """Tarayicida gosterilebilecek dosya yolunu dondurur.

    HEIC tarayicilarda native gosterilemedigi icin JPEG'e cevirip
    uploads/converted altinda onbelleklenir; diger formatlar oldugu gibi
    dondurulur (media_type None ise FileResponse dosya adindan tahmin eder).
    """
    path = Path(photo.storage_path)
    if path.suffix.lower() != ".heic":
        return path, None

    cached_path = CONVERTED_DIR / f"{photo.id}.jpg"
    if not cached_path.exists():
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.save(cached_path, format="JPEG", quality=90)

    return cached_path, "image/jpeg"

# --- Worker giris noktalari (is kuyrugu) -------------------------------
#
# Bu iki fonksiyon worker/main.py tarafindan, HICBIR acik transaction
# yokken cagrilir. Her biri KENDI DB oturumunu acar (istek oturumu coktan
# kapanmis olur) ve tamamen IDEMPOTENT'tir.
#
# TARIHSEL NOT (neden artik BackgroundTasks yok): VLM gorevleri onceden
# FastAPI BackgroundTasks ile tetikleniyordu; bu, senkron gorevleri
# anyio'nun PAYLASILAN thread havuzuna (CapacityLimiter(40)) gonderiyordu -
# yani senkron endpoint'lerin (yuz tespiti dahil) calistigi AYNI havuza.
# 404 fotograflik canli testte VLM gorevleri (~17 sn/foto) havuzu doldurup
# YENI yukleme isteklerini de bekletti (foto basina ~17 sn gecikme, toplam
# ~115 dk). Once ayri bir ThreadPoolExecutor'a tasindi (~50 kat iyilesme),
# simdi ise kalici kuyruga: surec yeniden baslasa bile is kaybolmuyor.


def run_face_pipeline_job(photo_id: uuid.UUID) -> None:
    """face_pipeline isinin worker giris noktasi.

    IDEMPOTENT: bu fotograf icin zaten yuz kaydi varsa hicbir sey yapmaz.
    detect_and_embed tum yuzleri tek commit'te yazdigi icin "DB'de yuz var"
    guvenilir bir "tamamlandi" sinyalidir. Bu kontrol tek basina bir
    check-then-act yarisina acikti (iki worker GERCEKTEN paralel calisirsa
    ikisi de "yok" gorebilir) - asagidaki photo-scoped advisory lock (PR-2)
    bu yarisi kapatir: kilit alinamadan detect_and_embed'e HIC girilmez.

    Bilinen sinir: cok nadir bir durumda (Qdrant upsert'i basarili, DB
    commit'i basarisiz) Qdrant'ta sahipsiz vektor kalabilir; DB tarafi geri
    alindigi icin is yeniden calisir ve dogru sonucu uretir, yalnizca
    kucuk bir Qdrant sizintisi olusur. (Advisory lock bunu COZMEZ - Postgres
    transaction'i ile Qdrant yazimi arasinda atomiklik yok, bkz. B.5 analiz
    notu; kilit sadece "ayni anda iki worker" durumunu engeller.)
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    lock_acquired = False
    try:
        photo = db.query(Photo).filter(Photo.id == photo_id).first()
        if photo is None:
            return

        lock_acquired = locks.acquire_photo_lock(db, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id)
        if not lock_acquired:
            raise LockConflict(f"face_pipeline: photo {photo_id} baska bir worker'da")

        already = db.query(Face).filter(Face.photo_id == photo_id).first()
        if already is not None:
            return  # zaten islenmis - at-least-once tekrari, no-op
        try:
            face_service.detect_and_embed(db, photo)
        except identity_locks.IdentityLockTimeout as exc:
            # Identity kilidi calismasi: baska bir islem (worker/HTTP) O AN
            # ayni kimligi tutuyor - GERCEK bir hata DEGIL. PR-2'nin
            # LockConflict'ine CEVIRIYORUZ ki worker/main.py'nin ZATEN VAR
            # OLAN cezasiz-requeue + esik-eskalasyon hatti (hicbir degisiklik
            # gerektirmeden) burada da calissin - bkz. identity_locks.py'deki
            # IdentityLockTimeout dokumentasyonu (neden burada cevriliyor,
            # db katmaninda degil).
            raise LockConflict(f"face_pipeline: identity kilidi mesgul ({exc})") from exc
        except identity_locks.StaleIdentityDecision as exc:
            # FAZ 1'in karari (bkz. face_service._apply_assignment_locked)
            # FAZ 2'ye gelindiginde artik gecersiz - araya bir merge/delete/
            # label girmis. GERCEK bir hata DEGIL, ayni sebeple (cember
            # import) LockConflict'e cevrilir - cezasiz requeue, worker
            # detect_and_embed'i BASTAN (yeni bir FAZ 1 karariyla) calistirir.
            #
            # AYRI bir log satiri (LockConflict'in KENDI genel mesajindan
            # farkli) - esik-eskalasyon tetiklendiginde operator "kaynak o an
            # mesguldu" (IdentityLockTimeout) ile "kimlik retire olmustu"
            # (StaleIdentityDecision) sebeplerini log'dan AYIRT edebilsin.
            logger.info(
                "face_pipeline: kimlik karari artik gecersiz (stale identity "
                "decision), cezasiz requeue ediliyor: %s", exc,
            )
            raise LockConflict(f"face_pipeline: kimlik karari artik gecersiz ({exc})") from exc
    finally:
        if lock_acquired:
            locks.release_photo_lock(db, locks.PHOTOAI_LOCK_CLASS_FACE, photo_id)
        db.close()


def run_vlm_analysis_job(photo_id: uuid.UUID) -> None:
    """vlm_analysis isinin worker giris noktasi.

    Bilincli olarak SENKRON: asyncio.run() bu worker thread'ine ait ayri
    bir olay dongusu acar, baska hicbir seyi etkilemez.

    PR-2: photo-scoped advisory lock (PHOTOAI_LOCK_CLASS_VLM) - face_pipeline
    ile AYNI photo_id icin bile birbirini BEKLEMEZ (farkli classid, iki is
    tipi tamamen bagimsiz tablolara yaziyor).
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    lock_acquired = False
    try:
        photo = db.query(Photo).filter(Photo.id == photo_id).first()
        if photo is None:
            return

        lock_acquired = locks.acquire_photo_lock(db, locks.PHOTOAI_LOCK_CLASS_VLM, photo_id)
        if not lock_acquired:
            raise LockConflict(f"vlm_analysis: photo {photo_id} baska bir worker'da")

        asyncio.run(run_vlm_analysis(db, photo))
    finally:
        if lock_acquired:
            locks.release_photo_lock(db, locks.PHOTOAI_LOCK_CLASS_VLM, photo_id)
        db.close()
