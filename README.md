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
```

`AI_PROVIDER=bedrock` seçilirse AWS kimlik bilgileri `.env`'de TUTULMAZ —
boto3'un standart kimlik bilgisi zinciri kullanılır (`~/.aws/credentials`,
ortam değişkenleri veya IAM rolü). Bkz. `AWS_REGION` / `AWS_BEDROCK_MODEL_ID`.

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

Fotoğraf yükleme artık **senkron değil**: `POST /photos` sadece dosyayı
kaydedip iki işi (`face_pipeline`, `vlm_analysis`) kuyruğa yazar ve `202`
döner. Bu işlerin gerçekten işlenmesi için ayrı worker süreçlerinin de
çalışıyor olması gerekir:

```bash
set JOB_TYPES=vlm_analysis  && python -m app.worker.main   # worker-vlm
set JOB_TYPES=face_pipeline && python -m app.worker.main   # worker-face
```

`JOB_TYPES` boş/geçersiz olursa worker **açılışta reddeder** (yanlış
yapılandırılmış bir worker'ın sessizce başka tipte iş çekmesini önlemek
için). Kaç worker süreci açılacağı `WORKER_VLM_PROCESSES` /
`WORKER_FACE_PROCESSES` ile kontrol edilir — `WORKER_FACE_PROCESSES` 1'in
üzerine çıkmadan önce `app/db/jobs_repository.py` başındaki not ve
`app/db/identity_locks.py` / `app/db/locks.py` okunmalı.

Kurulumun doğru gittiğini kontrol etmek için:

```bash
curl http://localhost:8001/health
```

`database`, `qdrant`, `vlm` bağımlılıklarının her biri ayrı ayrı raporlanır.

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
| `/health` | `main.py` | database/qdrant/vlm bağımsız sağlık kontrolü | - |

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
