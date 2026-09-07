# PhotoAI Backend

FastAPI tabanlı, kurumsal fotoğraf arşivi servisi. Yüklenen fotoğrafları bir
VLM (vision-language model) ile içerik açısından analiz eder, ayrı bir
yüz tanıma hattıyla fotoğraflardaki yüzleri tespit edip aynı kişiye ait
yüzleri kümeler/eşler. Her iki analiz de senkron HTTP isteği içinde DEĞİL,
PostgreSQL tabanlı bir **iş kuyruğu** üzerinden arka planda çalışan ayrı
worker süreçleriyle yürür.

## Özellikler

- Fotoğraf yükleme (JPEG, PNG, WEBP, HEIC) + içerik hash'iyle duplicate tespiti
- VLM ile içerik analizi: açıklama, ortam, aktivite, kişi sayısı, olası
  etkinlik, etiketler, tanınan kamusal figürler (`app/ai/dispatcher.py`)
  — sağlayıcı olarak yerel LM Studio **veya** AWS Bedrock seçilebilir
- Yüz tespiti (YuNet) + embedding (AuraFace) + HDBSCAN tabanlı otomatik
  kümeleme; kişi (`Person`) / küme (`Cluster`) etiketleme, birleştirme,
  yeniden atama uçları
- PostgreSQL tabanlı iş kuyruğu (`FOR UPDATE SKIP LOCKED`): her fotoğraf
  yüklemesi `face_pipeline` + `vlm_analysis` işlerini kuyruğa yazar, ayrı
  worker süreçleri bunları asenkron işler — yükleme isteği anında döner
- Çok kullanıcılı paylaşılan kurumsal arşiv: JWT tabanlı kimlik doğrulama
  (access + refresh token) ve rol tabanlı yetkilendirme (`admin`/`editor`/`viewer`)
- Aynı kimliğe (person/cluster) eşzamanlı erişimde deadlock/lost-update'i
  önleyen deterministik satır kilitleme ve kaynak-scoped advisory lock'lar

## Gereksinimler

- Python 3.11
- PostgreSQL
- Qdrant (yüz embedding'lerinin arama indeksi)
- VLM sağlayıcısı: yerel LM Studio **veya** AWS Bedrock erişimi

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Proje kökünde bir `.env` dosyası oluşturup değişkenleri tanımlayın — tam
liste ve açıklamalar için bkz. [.env.example](.env.example) ve
`app/core/settings.py`. Özet:

```
DATABASE_URL=postgresql://kullanici:sifre@localhost:5432/photoai_db
VLM_BASE_URL=http://localhost:1234/v1
VLM_MODEL=qwen/qwen2.5-vl-7b
AI_PROVIDER=lm_studio        # veya "bedrock"
JWT_SECRET=uzun-rastgele-bir-deger
CORS_ORIGINS=http://localhost:5173
```

`AI_PROVIDER=bedrock` seçilirse AWS kimlik bilgileri `.env`'de TUTULMAZ —
boto3'un standart kimlik bilgisi zinciri kullanılır (`~/.aws/credentials`,
ortam değişkenleri veya IAM rolü). Bkz. `AWS_REGION` / `AWS_BEDROCK_MODEL_ID`.

`CORS_ORIGINS` frontend'in servis edildiği origin(ler)i belirtir (virgülle
ayrılmış, `allow_credentials=True` olduğu için `*` joker karakteri
KULLANILAMAZ). Frontend backend ile aynı makinede değilse — ör. on-prem LAN
üzerinden başka bir cihazdan açılıyorsa — bkz. aşağıdaki
[LAN Üzerinden Erişim](#lan-üzerinden-erişim-on-prem).

## Veritabanı Migrasyonları

```bash
alembic upgrade head
```

## Kimlik Doğrulama

Sistem paylaşılan bir kurumsal arşiv olarak çalışır: tüm kullanıcılar aynı
fotoğraf/yüz/kişi verisini görür, kimin ne yaptığı ayrıca izlenir. Açık
self-register **yok** — yeni hesaplar sadece `admin` rolündeki bir kullanıcı
tarafından oluşturulur.

İlk admin hesabını oluşturmak için (migration sonrası, bir kereye mahsus):

```bash
venv\Scripts\python.exe scripts\create_admin.py --email admin@ornek.com --password guclu-bir-parola --display-name "Ad Soyad"
```

Roller: `admin` (kullanıcı yönetimi + tüm işlemler), `editor`
(yükleme/etiketleme/birleştirme/silme), `viewer` (salt okunur).

## Çalıştırma

Tüm servisleri (Qdrant, backend, worker'lar, frontend) tek seferde başlatmak
için repo kökündeki `start-photoai.bat` kullanılır (Docker **kullanılmıyor** —
her biri host üzerinde ayrı bir Windows süreci olarak çalışır):

```bash
..\start-photoai.bat
```

Backend tek başına:

```bash
uvicorn app.main:app --reload --reload-dir app --port 8001
```

> Not: backend varsayılan olarak **8001** portunda çalışır (8000 değil) —
> Windows'ta bazen port 8000'in "hayalet dinleyici" halinde takılı kalması
> (yeniden başlatmadan temizlenememesi) nedeniyle kalıcı olarak taşındı.
> Frontend'in `VITE_API_URL` değeri bununla eşleşmeli.

### LAN Üzerinden Erişim (on-prem)

Frontend backend ile **aynı makinede değilse** (ör. ofis ağındaki başka bir
bilgisayardan veya telefon/tablet'ten açılıyorsa), aşağıdaki **üçü birden**
gerekir — biri eksikse bağlantı sessizce başarısız olur ya da CORS hatası
verir:

1. **Backend tüm arayüzlerde dinlemeli** — varsayılan `uvicorn` yalnızca
   `127.0.0.1`'i dinler, bu LAN'daki başka bir cihazdan **erişilemez**.
   `--host 0.0.0.0` eklenmeli:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```
   `--reload` ile geliştirme modunda kullanıyorsanız:
   ```bash
   uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8001
   ```
   `start-photoai.bat` bunu **zaten içerir** — script'in backend komutu
   `--host 0.0.0.0` ile başlar, elle eklemeniz gerekmez. Yalnızca
   backend'i script dışında (`uvicorn ...` komutunu doğrudan) elle
   başlatıyorsanız bu bayrağı kendiniz eklemelisiniz.
2. **`CORS_ORIGINS` frontend'in gerçek LAN adresini içermeli** — backend'i
   çalıştıran makinenin LAN IP'sini `ipconfig` ile öğrenip frontend'in o
   adresten servis edildiğini varsayarak yazın:
   ```
   CORS_ORIGINS=http://192.168.1.50:5173,http://localhost:5173
   ```
3. **Frontend'in `VITE_API_URL`'i backend'in LAN IP'sine işaret etmeli** —
   `http://localhost:8001` değil, `http://192.168.1.50:8001` gibi.

Windows Güvenlik Duvarı ilk bağlantıda 8001 portu için izin isteyebilir —
"Özel ağlar" (Private networks) için izin verin.

Fotoğraf yükleme **üç aşamalı** ve hiçbir aşamada senkron AI işlemesi yok:

```
POST /photos ──► uploads/inbox/{id}.ext        (akış halinde yazım,
     │                 + {id}.ext.meta.json     .part → atomic rename)
     │           202 döner: dosya artık DİSKTE TAMAMLANMIŞ,
     │           tarayıcıdan bağımsız
     ▼
ingestion ────► photos + photo_exif + face_pipeline + vlm_analysis
(backend içi     TEK atomik commit; sonra dosya uploads/stored/'a taşınır
 arka plan       (sıra tersine: taşıma commit'ten ÖNCE, meta silme SONRA —
 döngüsü)         gerekçe: app/services/ingestion_service.py başlığı)
     ▼
worker'lar ───► yüz hattı / VLM / semantik indeks (DB kuyruğunu tüketir)
```

Ingestion döngüsü backend süreci içinde başlar (`app/main.py` lifespan) —
ayrı bir pencere gerekmez, backend her yeniden başladığında inbox taranır ve
yarım kalmış işlemler uzlaştırılır. `INGESTION_ENABLED=false` ile kapatılıp
`python -m app.ingestion.main` olarak ayrı süreçte de çalıştırılabilir.

**100 fotoğrafın tamamı beklenmez**: her fotoğraf kendi isteğinde
kalıcılaşır, ingestion onu tek tek işler ve worker'lar hemen alabilir —
upload ile işleme eşzamanlı ilerler.

Worker süreçlerinin **üçü de** ayrıca, ayrı pencerelerde çalışıyor olması
gerekir — biri eksikse ilgili iş tipi kuyrukta `queued` durumunda takılı
kalır:

```bash
set JOB_TYPES=vlm_analysis   && python -m app.worker.main   # worker-vlm
set JOB_TYPES=face_pipeline  && python -m app.worker.main   # worker-face
set JOB_TYPES=semantic_index && python -m app.worker.main   # worker-semantic
```

`JOB_TYPES` boş/geçersiz olursa worker **açılışta reddeder** (yanlış
yapılandırılmış bir worker'ın sessizce başka tipte iş çekmesini önlemek
için). Kaç worker süreci açılacağı `WORKER_VLM_PROCESSES` /
`WORKER_FACE_PROCESSES` ile kontrol edilir — `WORKER_FACE_PROCESSES` 1'in
üzerine çıkmadan önce `app/db/jobs_repository.py` başındaki not ve
`app/db/identity_locks.py` / `app/db/locks.py` okunmalı. `worker-semantic`
için ayrı bir süreç-sayısı ayarı yok (tek süreç yeterli — model küçük,
GPU'yu VLM ile paylaşmıyor).

`worker-semantic`, `vlm_analysis` işi **başarıyla** bittikten sonra
otomatik kuyruğa yazılan `semantic_index` işlerini tüketir; VLM analizi
JSON'unu embed edip Qdrant `photo_semantic` koleksiyonuna yazar
(`SEMANTIC_SEARCH_ENABLED=false` ise gerekmez).

Qdrant ve frontend'in ayrıca çalışıyor olması gerekir — bunlar bu repo'nun
parçası değildir; `..\start-photoai.bat` hepsini birlikte başlatır (bkz.
[Çalıştırma](#çalıştırma) üstü) ya da elle: Qdrant için `qdrant.exe`'yi
**kendi klasöründen** çalıştırın, frontend için `photoai-frontend`
dizininde `npm run dev`.

Kurulumun doğru gittiğini kontrol etmek için:

```bash
curl http://localhost:8001/health
```

`database`, `qdrant`, `vlm` bağımlılıklarının her biri ayrı ayrı raporlanır.
Ayrıca `workers` alanı her iş tipi (`face_pipeline`, `vlm_analysis`,
`semantic_index`) için worker sürecinin canlılığını gösterir:
`ok` (son ~15 sn içinde heartbeat), `stale` (kayıt var ama heartbeat eskimiş)
veya `down` (o tip için hiç worker başlamamış). Aynı bilgi
`GET /jobs/queue-status` yanıtındaki `workers` alanında da döner (Genel
Bakış paneli buradan okur — kuyruk boş olsa bile worker durumu görünür).

## API Uçları (özet)

Tam istek/cevap şemaları için servis ayaktayken `http://localhost:8001/docs`
(Swagger UI) kullanılması önerilir — burada sadece uç noktaların envanteri var.

| Prefix | Router | Açıklama | Auth |
|---|---|---|---|
| `/auth` | `auth.py` | login, refresh, logout, `/me` | - / Bearer |
| `/users` | `users.py` | kullanıcı oluşturma/listeleme/güncelleme | Bearer (`admin`) |
| `/photos` | `photos.py` | yükleme (`202`, asenkron), listeleme, tekil/toplu durum sorgusu, dosya/silme | Bearer |
| `/jobs` | `jobs.py` | tekil iş durumu, kuyruk durumu + ETA | Bearer |
| (prefix yok) | `faces.py` | kümeler, kişiler, etiketleme, birleştirme, yüz yeniden atama, birleştirme önerileri | Bearer |
| `/health` | `main.py` | database/qdrant/vlm bağımsız sağlık kontrolü + worker canlılığı (`workers`) | - |

## Proje Yapısı

```
app/
  main.py              # FastAPI uygulaması: router kaydı, CORS, global hata
                        # yakalayıcı, /health (db+qdrant+vlm bağımsız kontrol)
  schemas.py            # Pydantic istek/cevap şemaları

  ai/
    dispatcher.py        # VLM çağrı katmanı: AI_PROVIDER'a göre LM Studio
                          # (httpx, OpenAI-uyumlu) veya AWS Bedrock (boto3,
                          # Converse API) yolunu seçer; prompt, JSON şema
                          # doğrulama, dejenere-çıktı tespiti burada
    face_detector.py      # YuNet tabanlı yüz tespiti
    face_embedder.py      # AuraFace tabanlı yüz embedding çıkarımı

  core/
    settings.py           # pydantic-settings tabanlı tek merkez config (.env okur)
    security.py            # parola hash (bcrypt) + JWT access / opak refresh token
    dependencies.py        # FastAPI auth bağımlılıkları (get_current_user, require_role)
    exceptions.py           # ServiceError hata hiyerarşisi + tutarlı JSON gövdesi

  db/
    session.py             # SQLAlchemy engine/SessionLocal, get_db() dependency
    models.py               # ORM modelleri (User, Photo, PhotoAnalysis, Face,
                             # Person, Cluster, Job, UserJobCounter, ...)
    jobs_repository.py       # İş kuyruğu erişim katmanı: enqueue/claim_next/
                              # complete/fail/heartbeat/reap_stale, tek/çoklu
                              # job_type SQL dallanması (bkz. dosya başındaki not)
    locks.py                  # Kaynak-scoped PostgreSQL advisory lock'ları —
                               # aynı (job_type, photo_id) çiftinin iki worker
                               # tarafından gerçekten paralel işlenmesini önler
    identity_locks.py          # Person/Cluster satırlarını deterministik sırada
                                # kilitleyen yardımcı — çoklu kimlik güncelleyen
                                # akışlarda deadlock'u önler
    qdrant.py                  # Qdrant (yüz embedding arama indeksi) bağlantısı

  routers/                # İstek/cevap + yetkilendirme; iş mantığı YOK
    auth.py                 # /auth/login, /refresh, /logout, /me
    users.py                 # /users (admin-only kullanıcı yönetimi)
    photos.py                 # /photos (yükleme, listeleme, durum, dosya, silme)
    jobs.py                    # /jobs (tekil durum, kuyruk durumu + ETA)
    faces.py                    # kümeler, kişiler, etiketleme, birleştirme, yeniden atama

  services/                # İş mantığı (routers burayı çağırır)
    photo_service.py          # fotoğraf kaydetme (duplicate tespiti), worker
                               # entry point'leri (run_face_pipeline_job,
                               # run_vlm_analysis_job) — idempotent, kendi
                               # SessionLocal()'ını açar
    face_service.py            # yüz tespiti + hizalama + embedding orkestrasyonu,
                                # gerçek-zamanlı kimlik atama/bucket'lama
    candidate_search.py         # kimlik-havuzu aday arama soyutlaması (Qdrant'ı
                                 # face_service'ten ayırır)
    clustering_service.py        # HDBSCAN tabanlı toplu kümeleme + birleştirme önerileri
    person_service.py             # küme etiketleme, kişi birleştirme, yüz yeniden atama
    auth_service.py                # login, token yenileme/iptal, kullanıcı oluşturma

  worker/
    main.py                 # İş kuyruğu worker'i: JOB_TYPES doğrulama (fail-fast),
                             # claim → çalıştır → complete/fail döngüsü, heartbeat,
                             # reaper, graceful shutdown (SIGTERM/SIGINT)

  ingestion/
    main.py                 # Inbox izleme döngüsü: hızlı yoklama + periyodik
                             # uzlaştırma (crash kurtarma). Backend lifespan'inde
                             # arka plan thread'i olarak başlar.

alembic/
  versions/                # Veritabanı migrasyonları (kronolojik) — auth,
                            # yüz tanıma tabloları, iş kuyruğu, indeks/kolon eklemeleri

scripts/                  # Bakım/tek-seferlik araçlar (koddur, git'e girer)
  create_admin.py            # İlk admin hesabını oluşturur
  backup_system.py            # Yedekleme aracı
  backfill_person_cluster_centroids.py  # Geçmiş verilere centroid doldurma

tests/                    # pytest — repository/servis/worker davranışı,
                           # eşzamanlılık/kilit senaryoları, iş kuyruğu durum
                           # geçişleri; gerçek PostgreSQL'e karşı çalışır

.env.example              # Tüm ayarların şablonu (gerçek .env git'e girmez)
alembic.ini                # Alembic yapılandırması
requirements.txt            # pip freeze ile üretilmiş tam bağımlılık listesi
```

`uploads/`, `models/`, `reports/`, `logs/` klasörleri çalışma zamanı verisi/
çıktısıdır, koda dahil değildir ve `.gitignore` ile hariç tutulur (bkz.
o dosyadaki gerekçe notları).

`uploads/` içindeki yerleşim:

```
uploads/
├── inbox/       POST /photos'un yazdığı, henüz DB'ye alınmamış dosyalar
│                (+ .part = yazılıyor, + .meta.json = ingestion bitmedi)
├── stored/      ingestion tamamlanmış, kalıcı fotoğraflar
├── failed/      kalıcı olarak işlenemeyen dosyalar (elle inceleme için)
├── converted/   HEIC → JPG önizleme türevleri
└── {uuid}.ext   inbox mimarisinden ÖNCE yüklenmiş fotoğraflar —
                 taşınmadı, `Photo.storage_path` üzerinden çalışmaya devam eder
```

Inbox/ingestion durumu `GET /health` yanıtındaki `ingestion` alanında
görünür (`ready`, `in_progress`, `orphan_meta`, `failed`, işlenen sayıları,
son uzlaştırma zamanı).

## Bilinen Sınırlamalar

- `WORKER_FACE_PROCESSES` şu an **1** ile sınırlı tutuluyor: birden fazla
  yüz-hattı worker'ı aynı kimliğin (person/cluster) paylaşılan centroid
  durumunu eşzamanlı güncelleyebilir. `identity_locks.py`/`locks.py` bu
  yönde atılmış adımlar — 1'in üzerine çıkmadan önce ilgili testlerin
  (`test_identity_locks.py`, `test_face_pipeline_concurrency.py`,
  `test_worker_lock_escalation.py`) ve `jobs_repository.py` başındaki notun
  gözden geçirilmesi gerekir.
- `sequence_in_user` hiç sıfırlanmaz, `priority`'de yaşlanma/starvation
  önleme mekanizması yok — kabul edilmiş, bilinçli sınırlamalar.
