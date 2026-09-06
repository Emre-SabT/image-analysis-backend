import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile
from qdrant_client.models import PointStruct
from sqlalchemy.orm import Session
from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS
import pillow_heif

from app.core.settings import settings
from app.db import identity_locks, jobs_repository, locks, qdrant
from app.db.models import (
    JOB_TYPE_SEMANTIC_INDEX,
    ClusterConstraint,
    Face,
    Photo,
    PhotoAnalysis,
    PhotoExif,
)
from app.ai import dispatcher
from app.ai.dispatcher import analyze_photo
from app.services import activity_log_service, face_service, person_service, semantic_service

logger = logging.getLogger("photoai.photo_service")

pillow_heif.register_heif_opener()

# Depolama koku. PHOTOAI_UPLOAD_ROOT env degiskeni ile yonlendirilebilir -
# testler ve yan-yana calisan ikinci bir ornek, uretim uploads/ dizinini
# (ve onu izleyen canli ingestion dongusunu) kirletmesin diye. Verilmezse
# proje kokundeki "uploads".
UPLOAD_DIR = Path(os.environ.get("PHOTOAI_UPLOAD_ROOT", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONVERTED_DIR = UPLOAD_DIR / "converted"
CONVERTED_DIR.mkdir(exist_ok=True)

# --- Depolama yerlesimi (inbox -> stored) -------------------------------
#
#   uploads/
#   ├── inbox/      POST /photos'un yazdigi, HENUZ DB'ye alinmamis dosyalar
#   ├── stored/     ingestion tamamlanmis, kalici fotograflar
#   ├── failed/     kalici olarak islenemeyen dosyalar (elle inceleme icin)
#   ├── converted/  MEVCUT - HEIC->JPG onizleme turevleri (dokunulmadi)
#   └── {uuid}.ext  MEVCUT 483 fotograf - OLDUKLARI YERDE BIRAKILDI
#
# ESKI FOTOGRAFLAR TASINMADI (bilincli): her tuketici yolu Photo.storage_path
# uzerinden DB'den okuyor (face_service, dispatcher, system router, silme) -
# sabit kodlu dizin YOK. Dolayisiyla eski kayitlar "uploads/{uuid}.ext",
# yeniler "uploads/stored/{uuid}.ext" gosterir ve ikisi de calisir. Toplu
# dosya tasima + 483 satirlik UPDATE, hicbir sey kazandirmayan bir risktir.
INBOX_DIR = UPLOAD_DIR / "inbox"
STORED_DIR = UPLOAD_DIR / "stored"
FAILED_DIR = UPLOAD_DIR / "failed"
for _d in (INBOX_DIR, STORED_DIR, FAILED_DIR):
    _d.mkdir(exist_ok=True)

# Yaziliyor olan dosyanin uzantisi. Ingestion taramasi bu uzantiyi ATLAR -
# yarim dosyanin islenmesi boylece YAPISAL OLARAK imkansiz olur.
PART_SUFFIX = ".part"
# Yukleyen kullanici/orijinal dosya adi gibi, dosyanin KENDISINDE olmayan
# bilgiyi tasiyan yan dosya. Ingestion ayri bir surecte/thread'de calistigi
# icin HTTP istegindeki baglami baska turlu goremez.
META_SUFFIX = ".meta.json"
# Diski RAM'e almadan okumak icin parca boyutu (bkz. save_upload).
UPLOAD_CHUNK_BYTES = 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


class LockConflict(Exception):
    """PR-2: bu (is tipi, photo_id) kaynagini O AN baska bir worker
    isliyor (pg_try_advisory_lock False dondu). GERCEK bir hata DEGIL -
    handler hic calismadi. worker/main.py bunu jobs.fail() yerine
    jobs.requeue_lock_conflict() ile cezasiz kuyruga koyar (bkz.
    _process()'teki ayri except bloğu)."""


@dataclass(frozen=True)
class ReceivedUpload:
    """receive_upload'in sonucu.

    `duplicate=True` ise `existing` DOLU ve inbox'a HICBIR SEY birakilmamistir
    (gecici .part silinir) - is kaydi da olusmaz. `duplicate=False` ise dosya
    inbox'ta TAMAMLANMIS halde bekler; DB kaydini ve is kayitlarini ingestion
    dongusu olusturur (bkz. app/services/ingestion_service.py).
    """
    photo_id: uuid.UUID
    filename: str
    content_hash: str
    duplicate: bool
    existing: Photo | None = None


def receive_upload(
    db: Session, file: UploadFile, uploaded_by_user_id: uuid.UUID | None = None
) -> ReceivedUpload:
    """Yuklenen dosyayi inbox'a yazar. DB'ye HICBIR SEY YAZMAZ, is OLUSTURMAZ.

    ESKI DAVRANIS (save_upload) ile fark: eskiden bu fonksiyon Photo/EXIF
    satirlarini da olusturuyordu ve dosyanin TAMAMINI `file.file.read()` ile
    RAM'e aliyordu. Ikisi de degisti:

    1. AKIS HALINDE YAZIM: dosya 1 MB'lik parcalar halinde dogrudan diske
       yazilir ve SHA-256 ayni gecliste hesaplanir - ikinci bir okuma YOK,
       bellekte tam kopya YOK. 12 es zamanli 20 MB'lik yukleme eskiden ~240 MB
       anlik RAM demekti; simdi ~12 MB.

    2. ATOMIK GORUNURLUK: once "{photo_id}{ext}.part" adiyla yazilir, dosya
       TAMAMEN yazildiktan ve yan-dosya (meta) olustuktan SONRA os.replace ile
       nihai adina cevrilir. Ingestion taramasi .part uzantisini hic gormez -
       yarim dosyanin islenmesi YAPISAL OLARAK imkansizdir.

    YAN-DOSYA (.meta.json) NEDEN VAR: ingestion ayri bir dongude calisir ve
    HTTP istegindeki baglami (yukleyen kullanici, orijinal dosya adi)
    goremez; bu bilgi dosyanin kendisinde de yoktur. Meta, nihai yeniden
    adlandirmadan ONCE yazilir - boylece ingestion tamamlanmis bir goruntu
    dosyasini META'SIZ gorebilecegi bir an OLUSMAZ. Ayrica meta'nin VARLIGI
    "bu dosyanin ingestion'i henuz bitmedi" isaretidir (bkz. ingestion_service
    icindeki crash kurtarma).

    DUPLICATE: hash zaten akis sirasinda hesaplandigi icin ek maliyeti
    OLMADAN burada kontrol edilir - kullanicinin "Zaten arsivde" yanitini
    ANINDA almasi bu sayede korunur. Bu bir ON ELEME'dir; ASIL tekillik
    garantisi DB'deki UNIQUE(content_hash) kisitidir (bkz. models.py) -
    tam es zamanli iki ayni-dosya yuklemesinde ikisi de bu kontrolu gecebilir,
    o durumda ingestion tarafinda UNIQUE ihlali yakalanir.
    """
    ext = Path(file.filename).suffix.lower()
    photo_id = uuid.uuid4()
    part_path = INBOX_DIR / f"{photo_id}{ext}{PART_SUFFIX}"
    final_path = INBOX_DIR / f"{photo_id}{ext}"

    digest = hashlib.sha256()
    total_bytes = 0
    try:
        file.file.seek(0)
        with open(part_path, "wb") as out:
            while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
                digest.update(chunk)
                total_bytes += len(chunk)
                out.write(chunk)
        content_hash = digest.hexdigest()

        existing = db.query(Photo).filter(Photo.content_hash == content_hash).first()
        if existing is not None:
            # Yeni bir sey uretilmez: .part silinir, inbox'a hicbir sey
            # birakilmaz, is kuyruga alinmaz (eski davranisla ayni).
            part_path.unlink(missing_ok=True)
            logger.info("[UPLOAD] duplicate: %s (hash=%s)", file.filename, content_hash[:12])
            return ReceivedUpload(
                photo_id=existing.id, filename=file.filename,
                content_hash=content_hash, duplicate=True, existing=existing,
            )

        _write_inbox_meta(
            final_path,
            {
                "photo_id": str(photo_id),
                "original_filename": file.filename,
                "content_hash": content_hash,
                "size_bytes": total_bytes,
                "uploaded_by_user_id": str(uploaded_by_user_id) if uploaded_by_user_id else None,
                "received_at": datetime.utcnow().isoformat(),
            },
        )
        # ATOMIK: bu satirdan ONCE ingestion dosyayi goremez, SONRA tam
        # halini gorur. Ayni dosya sisteminde os.replace atomiktir.
        os.replace(part_path, final_path)
    except BaseException:
        # Yarim kalan .part'i birakma - aksi halde inbox'ta STALE_PART
        # suresince oksuz dosya bekler.
        part_path.unlink(missing_ok=True)
        _meta_path_for(final_path).unlink(missing_ok=True)
        raise

    logger.info(
        "[UPLOAD] inbox'a alindi: %s -> %s (%d bayt, hash=%s)",
        file.filename, final_path.name, total_bytes, content_hash[:12],
    )
    return ReceivedUpload(
        photo_id=photo_id, filename=file.filename,
        content_hash=content_hash, duplicate=False,
    )


def _meta_path_for(image_path: Path) -> Path:
    return image_path.with_name(image_path.name + META_SUFFIX)


def _write_inbox_meta(image_path: Path, meta: dict) -> None:
    """Yan-dosyayi KENDISI de atomik yazar: once .part, sonra rename.
    Yarim bir JSON, ingestion tarafinda ayristirma hatasi demekti."""
    meta_path = _meta_path_for(image_path)
    tmp = meta_path.with_name(meta_path.name + PART_SUFFIX)
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, meta_path)


def read_inbox_meta(image_path: Path) -> dict | None:
    """Yan-dosyayi okur. Yoksa/bozuksa None - cagiran bunu 'oksuz dosya'
    olarak ele alir (bkz. ingestion_service.ingest_file)."""
    meta_path = _meta_path_for(image_path)
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_photo_row(
    photo_id: uuid.UUID,
    filename: str,
    storage_path: Path,
    content_hash: str,
    uploaded_by_user_id: uuid.UUID | None,
    size_bytes: int,
    exif_source_path: Path,
) -> tuple[Photo, PhotoExif]:
    """Photo + PhotoExif nesnelerini uretir (session'a EKLEMEZ, commit ETMEZ).

    EXIF okumasi burada kalir: BACKEND_IHTIYACLARI.md #6 geregi tek seferlik
    ve senkron (Pillow'un okumasi ms mertebesinde, ayri bir arka plan isi
    GEREKTIRMEZ). Okuma basarisiz olursa alanlar None kalir ve ingestion YINE
    DE devam eder - EXIF ikincil bir zenginlestirme, fotografin sisteme
    alinmasini ENGELLEMEMELI (bkz. _extract_exif icindeki genis except).
    """
    photo = Photo(
        id=photo_id,
        filename=filename,
        storage_path=str(storage_path),
        status="processing",
        content_hash=content_hash,
        uploaded_by_user_id=uploaded_by_user_id,
    )
    exif_fields = _extract_exif(str(exif_source_path))
    exif = PhotoExif(photo_id=photo_id, file_size_bytes=size_bytes, **exif_fields)
    return photo, exif


def _clean_exif_str(value) -> str | None:
    """EXIF string alanlari bazen sonuna NUL/bosluk dolgusu ekler ("Canon\\x00\\x00")
    ya da tamamen bos gelir - ikisi de None'a normalize edilir (bos string
    frontend'de "var ama bos" gibi yanlis anlasilirdi)."""
    if value is None:
        return None
    cleaned = str(value).strip().strip("\x00").strip()
    return cleaned or None


def _dms_to_decimal(dms, ref) -> float | None:
    """EXIF GPS koordinatlari (derece, dakika, saniye) + yon referansi
    ("N"/"S"/"E"/"W") -> ondalik derece. Herhangi bir parca eksik/bozuksa
    None (UYDURULMAZ)."""
    if not dms or not ref:
        return None
    try:
        degrees, minutes, seconds = (float(x) for x in dms)
        decimal = degrees + minutes / 60 + seconds / 3600
        return -decimal if ref in ("S", "W") else decimal
    except (TypeError, ValueError):
        return None


def _extract_exif(image_path: str) -> dict:
    """Dosyadan GERCEKTEN okunabilen EXIF alanlarini cikarir - okunamayan/
    dosyada hic olmayan her alan None kalir, hicbir sey UYDURULMAZ (bkz.
    BACKEND_IHTIYACLARI.md #6). `file_size_bytes` burada YOK (cagiran zaten
    diske yazilan bayt sayisini biliyor, dosyayi ikinci kez stat() etmeye
    gerek yok).

    HEIC dahil TUM formatlar icin AYNI yol - `pillow_heif.register_heif_opener()`
    (modul yuklenirken bir kez cagriliyor) `Image.open()`'in HEIC dosyalarda
    da standart EXIF APP1 segmentini okumasini saglar, ayri bir dal GEREKMEZ.
    """
    result: dict = {
        "camera_make": None, "camera_model": None, "lens_model": None,
        "aperture": None, "shutter_speed": None, "iso": None,
        "focal_length": None, "captured_at": None,
        "gps_latitude": None, "gps_longitude": None, "copyright": None,
        "width_px": None, "height_px": None,
    }
    try:
        with Image.open(image_path) as img:
            result["width_px"], result["height_px"] = img.size

            exif = img.getexif()
            if not exif:
                return result
            tags = {TAGS.get(k, k): v for k, v in exif.items()}

            result["camera_make"] = _clean_exif_str(tags.get("Make"))
            result["camera_model"] = _clean_exif_str(tags.get("Model"))
            result["copyright"] = _clean_exif_str(tags.get("Copyright"))

            date_str = tags.get("DateTime")
            if date_str:
                try:
                    result["captured_at"] = datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass

            # Pozlama/lens bilgisi (FNumber, ExposureTime, ISO, LensModel,
            # FocalLength, DateTimeOriginal) ana IFD'de DEGIL, ayri bir
            # "Exif IFD" alt-blogunda (tag 0x8769) tutulur.
            exif_ifd = {}
            try:
                exif_ifd = {TAGS.get(k, k): v for k, v in exif.get_ifd(0x8769).items()}
            except (AttributeError, KeyError):
                pass

            dto = exif_ifd.get("DateTimeOriginal")
            if dto:
                try:
                    result["captured_at"] = datetime.strptime(str(dto), "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass

            f_number = exif_ifd.get("FNumber")
            if f_number:
                try:
                    result["aperture"] = f"f/{float(f_number):g}"
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            exposure = exif_ifd.get("ExposureTime")
            if exposure:
                try:
                    seconds = float(exposure)
                    result["shutter_speed"] = f"{seconds:g} sn" if seconds >= 1 else f"1/{round(1 / seconds)} sn"
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

            iso = exif_ifd.get("ISOSpeedRatings") or exif_ifd.get("PhotographicSensitivity")
            if iso is not None:
                try:
                    result["iso"] = int(iso[0] if isinstance(iso, (list, tuple)) else iso)
                except (TypeError, ValueError, IndexError):
                    pass

            focal = exif_ifd.get("FocalLength")
            if focal:
                try:
                    result["focal_length"] = f"{float(focal):g}mm"
                except (TypeError, ValueError):
                    pass

            result["lens_model"] = _clean_exif_str(exif_ifd.get("LensModel"))

            # GPS bilgisi de kendi alt-blogunda (tag 0x8825).
            try:
                gps_ifd = {GPSTAGS.get(k, k): v for k, v in exif.get_ifd(0x8825).items()}
                result["gps_latitude"] = _dms_to_decimal(gps_ifd.get("GPSLatitude"), gps_ifd.get("GPSLatitudeRef"))
                result["gps_longitude"] = _dms_to_decimal(gps_ifd.get("GPSLongitude"), gps_ifd.get("GPSLongitudeRef"))
            except (AttributeError, KeyError):
                pass
    except Exception as e:
        # EXIF okuma BASARISIZ olsa da yukleme akisi durmamali - ikincil bir
        # zenginlestirme (bkz. cagiran yerdeki yorum).
        logger.warning(f"EXIF okunamadi ({image_path}): {type(e).__name__}: {e}")
    return result




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
            model_name=dispatcher.CURRENT_MODEL_NAME,
            analyzed_at=datetime.utcnow(),
        )
        db.add(analysis)
        photo.status = "analyzed"
    except Exception as e:
        print(f"[ANALIZ HATASI] {photo.filename}: {type(e).__name__}: {e}")
        photo.status = "failed"

    db.commit()
    db.refresh(photo)

    # Analiz basariliysa semantik indeksleme isini kuyruga al - AYRI, ucuncu
    # bir hat (JOB_TYPE_SEMANTIC_INDEX). Bilincli olarak analiz commit'inden
    # SONRA ve AYRI bir commit'te: enqueue patlarsa analiz sonucu KAYBOLMAZ
    # (kısmi başarı ilkesi). Yukleyicisi bilinmeyen eski fotograflar sabit
    # sistem servis hesabina yazilir (settings.SYSTEM_USER_ID - is_active=False).
    if photo.status == "analyzed" and settings.SEMANTIC_SEARCH_ENABLED:
        try:
            user_id = photo.uploaded_by_user_id or uuid.UUID(settings.SYSTEM_USER_ID)
            jobs_repository.enqueue(
                db, JOB_TYPE_SEMANTIC_INDEX, {"photo_id": str(photo.id)}, user_id
            )
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[SEMANTIC ENQUEUE HATASI] {photo.filename}: {type(e).__name__}: {e}")

    return photo


def get_photo_with_analysis(db: Session, photo_id: uuid.UUID):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        return None, None
    analysis = db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == photo_id).first()
    return photo, analysis


def get_exif(db: Session, photo_id: uuid.UUID) -> PhotoExif | None:
    return db.query(PhotoExif).filter(PhotoExif.photo_id == photo_id).first()


def get_exif_map(db: Session, photo_ids: list[uuid.UUID]) -> dict[uuid.UUID, PhotoExif]:
    """N+1 onlemek icin TOPLU cekim - `_to_dict`'e `uploaded_by` icin
    kullanilan AYNI desen (bkz. routers/photos.py: list_photos)."""
    if not photo_ids:
        return {}
    rows = db.query(PhotoExif).filter(PhotoExif.photo_id.in_(photo_ids)).all()
    return {r.photo_id: r for r in rows}


def list_photos(db: Session):
    photos = db.query(Photo).order_by(Photo.created_at.desc()).all()
    out = []
    for p in photos:
        a = db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == p.id).first()
        out.append((p, a))
    return out


def delete_photo(db: Session, photo_id: uuid.UUID, actor_user_id: uuid.UUID | None = None) -> dict:
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
    # Silinmeden ONCE - ActivityLog'un target_label'i BUNA ihtiyac duyar
    # (bkz. asagidaki log() cagrisi, bulk .delete() sonrasi da erisilebilir
    # olsa da acik olsun diye ayri bir degiskene alindi).
    deleted_filename = photo.filename

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

    # Semantik arama indeksi: Qdrant 'photo_semantic' noktasi + photo_embeddings
    # satiri. photo_embeddings FK'si zaten CASCADE ama Qdrant noktasi CASCADE
    # ile GITMEZ - acikca silinmeli. Qdrant erisilemezse (ya da ozellik sonradan
    # kapatildiysa koleksiyon yoksa) fotograf silme akisi DURMAMALI - kirpim
    # dosyasi unlink hatasiyla ayni ilke ("DB temizligi yine de surmeli").
    if settings.SEMANTIC_SEARCH_ENABLED:
        try:
            semantic_service.remove_from_index(db, photo_id)
        except Exception as e:
            logger.warning(
                "photo_id=%s semantik indeksten silinemedi (%s: %s) - fotograf "
                "silme akisi suruyor, Qdrant'ta sahipsiz nokta kalabilir.",
                photo_id, type(e).__name__, e,
            )

    for path in (Path(photo.storage_path), CONVERTED_DIR / f"{photo.id}.jpg"):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    db.query(Photo).filter(Photo.id == photo_id).delete(synchronize_session=False)
    activity_log_service.log(
        db, actor_user_id, "photo_delete", "photo", photo_id, deleted_filename
    )
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


def run_semantic_index_job(photo_id: uuid.UUID) -> None:
    """semantic_index isinin worker giris noktasi.

    VLM JSON analizinden (photo_analysis) bir metin embedding'i uretip Qdrant
    'photo_semantic' koleksiyonuna + photo_embeddings izleme tablosuna yazar
    (bkz. semantic_service.index_photo).

    IDEMPOTENT: index_photo hem Qdrant upsert hem photo_embeddings ON CONFLICT
    kullanir - at-least-once tekrari zararsiz, ustune yazar.

    PR-2: photo-scoped advisory lock (PHOTOAI_LOCK_CLASS_SEMANTIC) - face/vlm
    ile AYNI photo_id icin bile birbirini BEKLEMEZ (farkli classid; uc is
    tipi bagimsiz tablolara yaziyor).

    photo_analysis HENUZ yoksa (index_photo False): analiz isi (vlm_analysis)
    henuz bitmemis demektir - LockConflict'e cevrilir; worker/main.py'nin
    VAR OLAN hatti cezasiz + backoff'lu requeue yapar. Normalde bu is zaten
    run_vlm_analysis BASARIYLA bittikten SONRA enqueue edilir (analiz coktan
    vardir); bu dal yalnizca at-least-once yeniden siralama / reaper
    senaryolari icin bir emniyet.
    """
    from app.db.session import SessionLocal

    if not settings.SEMANTIC_SEARCH_ENABLED:
        return  # ozellik kapali - is no-op olarak tamamlanir

    db = SessionLocal()
    lock_acquired = False
    try:
        photo = db.query(Photo).filter(Photo.id == photo_id).first()
        if photo is None:
            return

        lock_acquired = locks.acquire_photo_lock(
            db, locks.PHOTOAI_LOCK_CLASS_SEMANTIC, photo_id
        )
        if not lock_acquired:
            raise LockConflict(f"semantic_index: photo {photo_id} baska bir worker'da")

        if not semantic_service.index_photo(db, photo_id):
            raise LockConflict(
                f"semantic_index: photo {photo_id} icin VLM analizi henuz hazir degil"
            )
        db.commit()
    finally:
        if lock_acquired:
            locks.release_photo_lock(db, locks.PHOTOAI_LOCK_CLASS_SEMANTIC, photo_id)
        db.close()
