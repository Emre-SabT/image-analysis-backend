# ayarları tek yerden okuma (pydantic-settings tabanlı)
#
# app/config.py'nin yerini alır. pydantic-settings, .env dosyasını otomatik
# okur ve tipleri (bool/int/float) doğrular - eskiden her alan icin elle
# os.getenv(...).lower() == "true" gibi donusum yazmak gerekiyordu, artik
# gerekmiyor.

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    VLM_BASE_URL: str
    VLM_MODEL: str

    # "lm_studio" (yerel, OpenAI-uyumlu HTTP) | "bedrock" (AWS Bedrock, boto3)
    AI_PROVIDER: str = "lm_studio"

    # AWS Bedrock (AI_PROVIDER=bedrock iken kullanilir). Kimlik bilgileri
    # BURADA TUTULMAZ - boto3'un standart kimlik bilgisi zinciri (ortam
    # degiskenleri AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials,
    # ya da IAM rolu) kullanilir; AWS'nin kendi guvenlik pratigi budur.
    AWS_REGION: str = "us-east-1"
    AWS_BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Yuz tanima pipeline
    YUNET_MODEL_PATH: str = "models/face_detection_yunet_2023mar.onnx"
    AURAFACE_MODEL_DIR: str = "models/auraface"
    # 127.0.0.1, "localhost" DEGIL: Windows'ta "localhost" once IPv6'ya (::1)
    # cozuluyor, Qdrant ise 0.0.0.0 (IPv4) dinliyor. Basarisiz IPv6 denemesi
    # her cagriya ~2 sn ekliyordu.
    QDRANT_URL: str = "http://127.0.0.1:6333"
    QDRANT_API_KEY: str | None = None

    # Hibrit kimlik atama (opsiyonel katman, gercek-zamanli tek-esikli
    # atamanin ustune eklenir - bkz. face_service._assign_or_bucket).
    # Varsayilan KAPALI + GOLGE MOD: hicbir canli atama etkilenmez.
    HYBRID_ASSIGNMENT_ENABLED: bool = False
    HYBRID_SHADOW_MODE: bool = True
    HYBRID_CONFIDENCE_THRESHOLD: float = 0.55
    HYBRID_GAP_THRESHOLD: float = 0.05
    HYBRID_TOP_CANDIDATES: int = 5
    HYBRID_QUALITY_WEIGHTING_ENABLED: bool = True

    # --- Is kuyrugu / worker ---
    #
    # HICBIRI KODA GOMULU DEGIL: olculen ~437 foto/saat kapasitesi GELISTIRME
    # donanimindan (RTX 4050 Laptop, termal throttling) geliyor; sabit bir
    # kapasite degil. Musteri donanimina gore concurrency/replica ayarlanabilsin.
    #
    # JOB_TYPES: worker'in TUKETECEGI is tipleri (virgulle ayrilmis).
    # Varsayilani BILINCLI OLARAK BOS - "tumunu tuket" davranisi YOK.
    # Yanlis yapilandirilmis bir worker-face konteyneri sessizce VLM islerini
    # de cekerse tek GPU'daki eszamanlilik kontrolu coker; bu yuzden worker
    # bos/gecersiz JOB_TYPES ile BASLAMAYI REDDEDER (bkz. worker/main.py).
    JOB_TYPES: str = ""
    WORKER_CONCURRENCY: int = 1
    # Heartbeat, reaper timeout'unun BELIRGIN sekilde altinda olmali ki
    # calisan bir worker'in isi asla stale sayilmasin (30 sn << 300 sn).
    JOB_HEARTBEAT_SECONDS: int = 30
    JOB_STALE_TIMEOUT_SECONDS: int = 300
    JOB_POLL_INTERVAL_SECONDS: float = 1.0
    JOB_REAP_INTERVAL_SECONDS: int = 60
    # PR-2 (photo-scoped advisory lock, henuz uygulanmadi) icin taban
    # bekleme suresi: baska bir worker ayni photo_id/is tipini O ANDA
    # isliyorsa is bu kadar + jitter sonra tekrar denenir. BILINCLI
    # OLARAK JOB_STALE_TIMEOUT_SECONDS'TAN BAGIMSIZ - bu bir "worker
    # oldu mu" karari degil, "kaynak o an mesguldu" bilgisi, cok daha
    # kisa bir bekleme yeterli.
    JOB_LOCK_CONFLICT_BACKOFF_SECONDS: int = 20
    # Bir is bu kadar (veya daha fazla) kez lock-conflict requeue'su
    # yasarsa (job.attempts - SADECE claim sayisi, failure_count DEGIL,
    # bkz. jobs_repository.py basindaki not), bu ARTIK NORMAL gecici
    # cakisma degil, muhtemelen SIZMIS bir advisory lock (worker crash
    # sonrasi pool'a lock tutan bir baglanti donmus olabilir) sinyalidir.
    # requeue_lock_conflict() BILINCLI OLARAK hicbir ust sinira sahip
    # DEGIL (failure_count'a dokunmuyor - bkz. PR-1) - bu yuzden esik
    # asilinca job OTOMATIK fail EDILMEZ (bu, kaybi/manuel mudahaleyi
    # gerektirir), sadece logger.error ile ESKALE EDILIR: baglanti
    # havuzundaki sizinti tipik olarak worker process'i yeniden
    # baslatildiginda kendiliginden duzelir (advisory lock, backend
    # session'i sonlaninca da duser), bu yuzden gozlemlenebilirlik +
    # operasyonel mudahale (worker restart / pg_locks incelemesi) daha
    # guvenli bir varsayilan. 10, ~JOB_LOCK_CONFLICT_BACKOFF_SECONDS
    # araligiyla carpildiginda (~200-300 sn) JOB_STALE_TIMEOUT_SECONDS
    # (300) ile ayni buyuklukte bir pencereye denk gelecek sekilde
    # secildi: bu sureden uzun surmesi, "gecici cakisma" aciklamasinin
    # artik makul olmadigi anlamina gelir.
    JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS: int = 10

    # --- Kimlik (person/cluster) centroid arama backend'i ---
    #
    # 'qdrant'        : bugunku davranis - IDENTITY_POOL_COLLECTION'da HNSW
    #                   arama (QdrantCandidateFinder). Centroid PG'de de
    #                   TUTULUYOR (dual-write) ama okunmuyor.
    # 'pg_brute_force': persons/clusters.centroid'ten dogrudan brute-force
    #                   arama (PgBruteForceCandidateFinder) - Qdrant
    #                   identity_pool'a artik BAGIMLI DEGIL.
    #
    # Cutover BILINCLI OLARAK ayri bir bayrak: yazma yolu (dual-write, HER
    # ZAMAN acik) ile okuma yolu (bu bayrak) birbirinden bagimsiz - worker
    # durdurmadan, istenen zaman TEK ayar degisikligiyle cevrilir. Bakim
    # penceresi SADECE ilk backfill icin gerekir (bkz.
    # scripts/backfill_person_cluster_centroids.py), okuma gecisi icin
    # DEGIL.
    #
    # DIKKAT - 'pg_brute_force' PR-D BITENE KADAR SET EDILEMEZ (asagidaki
    # validator BUNU UYGULAMA BASLANGICINDA REDDEDER, calisma zamaninda
    # DEGIL): merge_identities/label_cluster/reassign_face/delete_photo
    # (person_service.py, photo_service.delete_photo) HALA SADECE Qdrant'a
    # yaziyor - bu islemler sirasinda persons.centroid/clusters.centroid
    # GUNCELLENMIYOR. Bayrak erken acilirsa: bir admin iki kimligi merge
    # ederse, PG'deki (bayat) iki AYRI centroid hala aktif kimlikmis gibi
    # aranir - merge edilmis kimlikler PG brute-force aramasinda YENIDEN
    # AYRISIR. PR-D bu dort fonksiyonu da PG centroid'i (identity_locks.
    # lock_identities ile) guncelleyecek sekilde tamamlayinca bu kisit
    # KALDIRILACAK.
    IDENTITY_SEARCH_BACKEND: str = "qdrant"

    @field_validator("IDENTITY_SEARCH_BACKEND")
    @classmethod
    def _validate_identity_search_backend(cls, v: str) -> str:
        if v not in ("qdrant", "pg_brute_force"):
            raise ValueError(f"Gecersiz IDENTITY_SEARCH_BACKEND: {v!r} (qdrant|pg_brute_force)")
        if v == "pg_brute_force":
            raise ValueError(
                "IDENTITY_SEARCH_BACKEND=pg_brute_force PR-D bitene kadar ACILAMAZ - "
                "yukaridaki 'DIKKAT' notuna bakin (merge/label/reassign/delete_photo "
                "PG centroid'i henuz guncellemiyor)."
            )
        return v
    # queue-status tahmini suresi bu kadar tamamlanmis isin ORTALAMASINDAN
    # hesaplanir (sabit bir tahmin degeri kullanilmaz).
    JOB_ETA_SAMPLE_SIZE: int = 20
    # Kac worker SURECI baslatilacagi (start-photoai.bat bu degerleri okur).
    # Docker kullanilmiyor - worker'lar host'ta ayri Python surecleri olarak
    # calisir; eszamanlilik bu sayilarla kontrol edilir.
    #
    # VLM icin varsayilan 1: tek GPU paylasiliyor, ikinci bir surec VRAM'i
    # tasirir (olculdu: model tek basina 5,6 GB / 6,1 GB VRAM kullaniyor).
    WORKER_VLM_PROCESSES: int = 1
    # DIKKAT: 1'den buyuk yapmadan ONCE identity_pool centroid lost-update
    # sorunu cozulmeli - bkz. jobs_repository.py basindaki TODO.
    WORKER_FACE_PROCESSES: int = 1

    # Kimlik dogrulama (JWT). SERVICE_KEY kaldirildi - coklu kullanicili
    # sisteme gecisle birlikte her istemci kendi hesabiyla giris yapiyor.
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30


settings = Settings()
