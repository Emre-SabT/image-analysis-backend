# tablolar

import uuid
from datetime import datetime
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, func, text,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from app.db.session import Base


# --- Coklu kullanici / kimlik dogrulama ---
#
# Paylasilan kurumsal arsiv modeli: tum kullanicilar ayni foto/yuz/kisi
# verisini gorur ve yonetir (veri izolasyonu yok). Bu tablolar sadece KIM
# oturum actigini ve KIM neyi olusturdugunu takip eder.

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    role = Column(String, nullable=False, server_default="viewer")  # admin | editor | viewer
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(DateTime, default=datetime.utcnow)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    # Ham token DB'de hicbir zaman tutulmaz - sadece SHA-256 hash'i. DB
    # sizarsa token'lar dogrudan kullanilamaz.
    token_hash = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Photo(Base):
    __tablename__ = "photos"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    status = Column(String, default="uploaded")  # uploaded -> processing -> analyzed -> failed
    # Dosya tekillestirme: yuklenen dosyanin SHA-256'si (photo_service.save_upload).
    # Nullable: migration'dan once yuklenmis eski kayitlar scripts/backfill_content_hash.py
    # calistirilana kadar NULL kalir - o ana kadar duplicate kontrolune dahil olmazlar.
    #
    # UNIQUE DEGIL (bilincli): backfill sirasinda bu ozellik eklenmeden once
    # yuklenmis 7 grup halihazirda yinelenen fotograf bulundu (ayni icerik,
    # farkli klasor/isimle iki kez yuklenmis). Unique kisit bu satirlarin
    # hash'ini doldururken commit'i patlatirdi. Tekillestirme uygulama
    # katmaninda (save_upload) zaten yapiliyor; DB'de sadece sorgu icin
    # (WHERE content_hash = ...) duz bir index yeterli.
    content_hash = Column(String(64), nullable=True, index=True)
    # Coklu kullanici gecisinden ONCE yuklenmis fotograflarda NULL kalir -
    # sahte sahiplik uydurulmadi (bkz. migration c1e9a2f6b3d4).
    uploaded_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PhotoExif(Base):
    """BACKEND_IHTIYACLARI.md #6 - dosyadan GERCEKTEN okunan EXIF/teknik
    meta veri. Yukleme sirasinda BIR KEZ, senkron doldurulur
    (photo_service.save_upload -> _extract_exif) - VLM/yuz hatti gibi ayri
    bir arka plan isi GEREKMEZ, Pillow'un EXIF okumasi cok ucuzdur.
    Okunamayan/dosyada hic olmayan alanlar NULL kalir, UYDURULMAZ."""
    __tablename__ = "photo_exif"
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.id", ondelete="CASCADE"), primary_key=True)
    camera_make = Column(String, nullable=True)
    camera_model = Column(String, nullable=True)
    lens_model = Column(String, nullable=True)
    aperture = Column(String, nullable=True)  # "f/2.8"
    shutter_speed = Column(String, nullable=True)  # "1/250 sn"
    iso = Column(Integer, nullable=True)
    focal_length = Column(String, nullable=True)  # "50mm"
    # GERCEK cekim tarihi (EXIF DateTimeOriginal) - Photo.created_at
    # (YUKLEME zamani) ile KARISTIRILMAZ, ikisi FARKLI olaylar.
    captured_at = Column(DateTime, nullable=True)
    gps_latitude = Column(Float, nullable=True)
    gps_longitude = Column(Float, nullable=True)
    copyright = Column(String, nullable=True)
    width_px = Column(Integer, nullable=True)
    height_px = Column(Integer, nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)


class PhotoAnalysis(Base):
    __tablename__ = "photo_analysis"
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.id"), primary_key=True)
    description = Column(Text)
    primary_object = Column(String)
    secondary_objects = Column(ARRAY(String), server_default=text("'{}'"))
    environment = Column(ARRAY(String), server_default=text("'{}'"))  # mekan/ortam etiketleri
    attributes = Column(ARRAY(String), server_default=text("'{}'"))
    action = Column(String)
    mood = Column(String)
    use_case = Column(String)
    context = Column(ARRAY(String), server_default=text("'{}'"))
    style = Column(ARRAY(String), server_default=text("'{}'"))
    audience = Column(ARRAY(String), server_default=text("'{}'"))
    public_figures = Column(JSONB, server_default=text("'[]'::jsonb"))
    all_tags = Column(ARRAY(String), server_default=text("'{}'"))
    model_name = Column(String)
    analyzed_at = Column(DateTime)


# --- Yuz tanima pipeline (Teknik Tasarim Dokumani Bolum 11) ---
#
# Embedding'lerin kendisi burada YOK: 512-d vektorler Qdrant'ta tutulur
# (point_id = faces.id, koleksiyon adlari app/db/qdrant.py'de). Bu tablolar
# yalnizca PostgreSQL'in kaynak-of-truth oldugu iliskisel/meta veriyi tasir.

class Cluster(Base):
    __tablename__ = "clusters"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String, default="unlabeled")  # unlabeled | labeled | merged
    size = Column(Integer, default=0)
    centroid_updated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # PR-A (identity/centroid eszamanlilik calismasi): PG'de OTORITER
    # centroid kopyasinin altyapisi - bkz. migration a7b3c9d1e5f2. Bu asamada
    # HICBIR OKUYUCU bunu kullanmiyor (PR-C'ye kadar), sadece backfill
    # tarafindan doldurulur. NULL = henuz backfill edilmemis/uye yok.
    centroid = Column(LargeBinary, nullable=True)


class Person(Base):
    __tablename__ = "persons"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    display_name = Column(String, nullable=False)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"), nullable=True)
    face_count = Column(Integer, default=0)
    # Eskiden serbest metin (istemcinin gonderdigi herhangi bir string,
    # dogrulanmiyordu). Simdi gercek kullanici FK'si - deger artik
    # get_current_user'dan geliyor, istemci govdesinden degil.
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)  # FR-13: kisi bazli silme (soft delete)
    # PR-A - bkz. Cluster.centroid yorumu, ayni gerekce.
    centroid = Column(LargeBinary, nullable=True)
    centroid_updated_at = Column(DateTime(timezone=True), nullable=True)


class Face(Base):
    __tablename__ = "faces"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.id", ondelete="CASCADE"), nullable=False)
    bbox = Column(JSONB, nullable=False)  # {x, y, w, h}
    landmarks = Column(JSONB, nullable=False)  # 5 nokta: [[x,y], ...]
    det_confidence = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=True)
    crop_path = Column(String, nullable=False)  # yerel disk: uploads/faces/{face_id}.jpg
    # Bolum 8.1: alani goruntunun %0.1'inden kucuk yuzler "arka plan yuzu" olarak
    # isaretlenir ve kumelemeye dusuk oncelikle girer (silinmez).
    is_background = Column(Boolean, nullable=False, server_default=text("false"))
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"), nullable=True)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    assigned_by = Column(String, nullable=True)  # auto | human | suggestion
    created_at = Column(DateTime, default=datetime.utcnow)


class ClusterConstraint(Base):
    __tablename__ = "cluster_constraints"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    face_id_a = Column(UUID(as_uuid=True), ForeignKey("faces.id"), nullable=False)
    face_id_b = Column(UUID(as_uuid=True), ForeignKey("faces.id"), nullable=False)
    type = Column(String, nullable=False)  # must_link | cannot_link
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # HANGI eylemden geldigini ayirt eder - `type` tek basina yetmiyor:
    # "cannot_link" hem bir birlestirme ONERISININ REDDinden (reject_merge)
    # HEM DE alakasiz bir "kimlikten ayir" (reassign_face split) eyleminden
    # yazilabiliyor. GET /identities/merge-history bu ikisini KARISTIRMAMAK
    # icin source'a gore filtreler (bkz. BACKEND_IHTIYACLARI.md).
    # merge_accept | merge_reject | face_detach | NULL (bu kolon eklenmeden
    # ONCE yazilmis eski kayitlar - kaynagi GERCEKTEN bilinmiyor, UYDURULMAZ).
    source = Column(String, nullable=True)


# --- Etkinlik/audit gunlugu (BACKEND_IHTIYACLARI.md #5 + kullanici istegi) --
#
# Kurumsal ORTAK havuz: bir albumde/kimlikte/fotografta yapilan HER islemden
# (silme, birlestirme, albume ekleme/cikarma, yeniden adlandirma, yuz
# atama...) TUM kullanicilarin haberi olmali. Bu tablo, ilgili servis
# fonksiyonu her mutasyon yaptiginda AYNI transaction icinde (commit'ten
# ONCE) bir satir yazar - bkz. app/services/activity_log_service.py.

class ActivityLog(Base):
    __tablename__ = "activity_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # SET NULL (photos.uploaded_by_user_id/persons.created_by_user_id ile AYNI
    # ilke, bkz. migration b1c4d6e8f0a2) - bir kullanici kalici silinse bile
    # gecmis islem kaydi KAYBOLMAZ, yalnizca aktoru bilinmez hale gelir.
    actor_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    # photo_upload | photo_delete | album_create | album_delete |
    # album_photo_add | album_photo_remove | person_named | person_renamed |
    # identity_merge | identity_reject_merge | identity_delete | face_reassign
    action = Column(String, nullable=False)
    target_kind = Column(String, nullable=False)  # photo | album | person | cluster | face
    # FK DEGIL (bilincli): hedef sonradan silinebilir, o zaman bile GECMIS
    # kaydin kendisi anlamli kalmali (bkz. target_label).
    target_id = Column(UUID(as_uuid=True), nullable=True)
    # Insan-okunabilir etiket KARAR ANINDA donduruldu (ör. dosya adi, album
    # adi, kisi adi) - hedef sonradan silinse/yeniden adlandirilsa bile
    # GECMISTEKI olay anlamli kalir, "bilinmeyen ID" gorunmez.
    target_label = Column(String, nullable=True)
    # Ek baglam (ör. {"added": 5, "album_name": "..."}) - opsiyonel, aksiyon
    # turune gore degisir. Python tarafinda `extra` (SQLAlchemy'de `metadata`
    # Base'in kendi ozniteligiyle CAKISIR).
    extra = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# --- Albumler (BACKEND_IHTIYACLARI.md #1) ---
#
# Bir fotograf BIRDEN FAZLA albumde olabilir - bu yuzden `Photo.album_id`
# DEGIL, ayri bir join tablosu (`AlbumPhoto`). Album silinince fotograflar
# SILINMEZ (yalnizca album_photos kayitlari CASCADE ile gider) - kisi/kume
# silmedeki "fotograflara dokunmaz" ilkesiyle AYNI (bkz. person_service.
# delete_identity).

class Album(Base):
    __tablename__ = "albums"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Liste/kart görünümünde thumbnail için - kullanıcı seçer (varsayılan:
    # albüme eklenen İLK fotoğraf, bkz. album_service). Fotoğraf silinirse
    # SET NULL (migration) - albüm kapaksız kalır, hiçbir şey patlamaz.
    cover_photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.id"), nullable=True)


class AlbumPhoto(Base):
    __tablename__ = "album_photos"
    album_id = Column(UUID(as_uuid=True), ForeignKey("albums.id", ondelete="CASCADE"), primary_key=True)
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.id", ondelete="CASCADE"), primary_key=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    added_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)


# --- Is kuyrugu (PostgreSQL tabanli, FOR UPDATE SKIP LOCKED) ---
#
# Redis/Celery yerine PostgreSQL secildi: urun on-premise kuruluyor ve
# PostgreSQL zaten kurulum listesinde; Redis musteri sahasinda ayri bir
# kurulum/izleme/yedekleme yuku demek olurdu. Olculen VLM kapasitesi
# ~0,12 is/sn iken SKIP LOCKED saniyede yuzlerce isi kaldiriyor - kuyruk
# teknolojisi bu sistemde darbogaz degil.

JOB_TYPE_FACE_PIPELINE = "face_pipeline"
JOB_TYPE_VLM_ANALYSIS = "vlm_analysis"

# Worker'in JOB_TYPES dogrulamasi bu kumeye karsi yapilir (fail-fast).
KNOWN_JOB_TYPES = frozenset({JOB_TYPE_FACE_PIPELINE, JOB_TYPE_VLM_ANALYSIS})

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_DONE = "done"
JOB_STATUS_FAILED = "failed"


class Job(Base):
    """Koordinasyon tablosu - VERI DEPOSU DEGIL.

    Bilincli olarak `result`/`output` kolonu YOK: analiz ciktisi worker
    tarafindan ilgili veri tablosuna (photo_analysis / faces) ayrica
    yazilir. jobs yalnizca "bu is kimde, hangi asamada" sorusunu yanitlar.

    Zaman kolonlari TIMESTAMPTZ (mevcut tablolardaki naive DateTime'dan
    farkli): kuyrukta now() karsilastirmalari ve cok-worker'li koordinasyon
    var, timezone-aware olmak burada gercek bir dogruluk meselesi.
    """

    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String(50), nullable=False)  # KNOWN_JOB_TYPES
    payload = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    status = Column(String(20), nullable=False, server_default=JOB_STATUS_QUEUED)
    priority = Column(Integer, nullable=False, server_default=text("0"))
    # NOT NULL: her iki job tipi de kimlik dogrulamali POST /photos'tan
    # doguyor, ayrica adil siralama her isin bir kullanici kuyruguna ait
    # olmasina dayaniyor.
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    # Kullanici basina monoton artan sira no - bkz. UserJobCounter.
    sequence_in_user = Column(BigInteger, nullable=False)
    attempts = Column(Integer, nullable=False, server_default=text("0"))
    # "kac kez GERCEKTEN denenip basarisiz oldu" - attempts'ten (kac kez
    # claim edildi) BILINCLI OLARAK AYRI (bkz. migration f1a2b3c4d5e6).
    # SADECE jobs_repository.fail() cagrildiginda artar; reaper'in yanlis-
    # pozitif reclaim'i ya da lock-cakismasi requeue'su (PR-2,
    # requeue_lock_conflict) buna DOKUNMAZ - gercek bir deneme degildir.
    failure_count = Column(Integer, nullable=False, server_default=text("0"))
    max_attempts = Column(Integer, nullable=False, server_default=text("3"))
    locked_by = Column(String(100), nullable=True)
    # Heartbeat bu alani tazeler; reaper bunun eskiligine bakar.
    locked_at = Column(DateTime(timezone=True), nullable=True)
    run_after = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)


class UserJobCounter(Base):
    """Kullanici basina sira sayaci.

    COUNT(*) BILINCLI OLARAK KULLANILMIYOR: yaris durumuna acik (iki
    es zamanli enqueue ayni sayiyi okuyup ayni sequence'i yazabilir).
    Bunun yerine tek ifadelik atomik artirim kullaniliyor
    (INSERT ... ON CONFLICT DO UPDATE ... RETURNING).
    """

    __tablename__ = "user_job_counters"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    next_sequence = Column(BigInteger, nullable=False, server_default=text("0"))
