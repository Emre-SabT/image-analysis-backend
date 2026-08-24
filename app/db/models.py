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

class PhotoAnalysis(Base):
    __tablename__ = "photo_analysis"
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.id"), primary_key=True)
    description = Column(Text)
    environment_type = Column(String)  # indoor | outdoor | mixed
    people_count = Column(Integer)
    possible_event = Column(String)
    primary_object = Column(String)
    secondary_objects = Column(ARRAY(String), server_default=text("'{}'"))
    environment = Column(ARRAY(String), server_default=text("'{}'"))  # mekan/ortam etiketleri (eski environment_type ile karistirma)
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
