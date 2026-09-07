# PhotoAI Backend — Kurulum ve Çalıştırma Rehberi

> Bu rehber, projeyi hiç görmemiş bir geliştiricinin **sıfır bir Windows makinesinde**
> backend'i kurup çalıştırabilmesi için hazırlanmıştır. Buradaki her sürüm, dosya yolu,
> ortam değişkeni ve komut, repository'deki gerçek koddan (`requirements.txt`,
> `app/core/settings.py`, `app/worker/main.py`, `alembic/`, `.env.example`) doğrulanmıştır.
>
> **Referans commit:** `cd09a73` — *feat: yapılandırılabilir CORS + asenkron upload/ingestion hattı + worker gözlemi*

---

## İçindekiler

1. [Sistem Gereksinimleri](#1-sistem-gereksinimleri)
2. [Kullanılan Teknolojiler](#2-kullanılan-teknolojiler)
3. [Mimari Özet — Neyi Neden Kuruyoruz](#3-mimari-özet--neyi-neden-kuruyoruz)
4. [Adım 1 — Ön Koşul Yazılımların Kurulumu](#adım-1--ön-koşul-yazılımların-kurulumu)
5. [Adım 2 — Repository ve Python Sanal Ortamı](#adım-2--repository-ve-python-sanal-ortamı)
6. [Adım 3 — Bağımlılıkların Kurulumu (PyTorch dikkat!)](#adım-3--bağımlılıkların-kurulumu-pytorch-dikkat)
7. [Adım 4 — PostgreSQL Veritabanının Hazırlanması](#adım-4--postgresql-veritabanının-hazırlanması)
8. [Adım 5 — Qdrant Vektör Veritabanı](#adım-5--qdrant-vektör-veritabanı)
9. [Adım 6 — AI/ML Model Dosyalarının İndirilmesi](#adım-6--aiml-model-dosyalarının-indirilmesi)
10. [Adım 7 — VLM Sağlayıcısı (LM Studio veya AWS Bedrock)](#adım-7--vlm-sağlayıcısı-lm-studio-veya-aws-bedrock)
11. [Adım 8 — `.env` Dosyasının Oluşturulması](#adım-8--env-dosyasının-oluşturulması)
12. [Adım 9 — Alembic Migration'ları](#adım-9--alembic-migrationları)
13. [Adım 10 — İlk Admin Hesabı](#adım-10--i̇lk-admin-hesabı)
14. [Adım 11 — Backend'i Başlatma](#adım-11--backendi-başlatma)
15. [Adım 12 — Worker Süreçlerini Başlatma](#adım-12--worker-süreçlerini-başlatma)
16. [Adım 13 — Kurulumun Doğrulanması](#adım-13--kurulumun-doğrulanması)
17. [Adım 14 — Testlerin Çalıştırılması (opsiyonel)](#adım-14--testlerin-çalıştırılması-opsiyonel)
18. [Tüm Servisleri Tek Komutla Başlatma](#tüm-servisleri-tek-komutla-başlatma)
19. [Ortam Değişkenleri — Tam Referans](#ortam-değişkenleri--tam-referans)
20. [Kurulum Checklist'i](#kurulum-checklisti)
21. [Yaygın Hatalar ve Çözümleri](#yaygın-hatalar-ve-çözümleri)

---

## 1. Sistem Gereksinimleri

### 1.1 Donanım

| Bileşen | Minimum | Önerilen | Gerekçe (koddan) |
|---|---|---|---|
| **CPU** | 4 çekirdek x64 | 8 çekirdek | Yüz tespiti (YuNet) ve yüz embedding (AuraFace) **bilinçli olarak CPU'ya sabitlenmiştir** — `app/ai/face_detector.py` `DNN_TARGET_CPU`, `app/ai/face_embedder.py` `CPUExecutionProvider` / `ctx_id=-1`. |
| **RAM** | 8 GB | 16 GB | PostgreSQL + Qdrant + backend + 3 worker süreci aynı makinede. |
| **GPU** | Yok (CPU modu mümkün) | NVIDIA, **CUDA 12.1 uyumlu**, ≥ 6 GB VRAM | `requirements.txt` içinde `torch==2.5.1+cu121`; `EMBEDDING_DEVICE` varsayılanı `cuda`. |
| **VRAM** | — | ≥ 6 GB | `settings.py` ölçüm notu: yerel VLM (Qwen2.5-VL-7B) tek başına **5,6 GB / 6,1 GB VRAM** kullanıyor. e5-base embedding modeli ayrıca **~1,1 GB** (`app/ai/embedding/__init__.py`). |
| **Disk** | 10 GB boş | 50 GB+ | Model dosyaları ~1,4 GB + `venv/` (PyTorch CUDA ile ~4–5 GB) + `uploads/` (fotoğraf arşivi, sınırsız büyür). |

> **GPU yoksa ne olur?** Sistem çalışır. `EMBEDDING_DEVICE=cpu` yapıp VLM için
> `AI_PROVIDER=bedrock` (bulut) seçmeniz yeterlidir. Yüz hattı zaten CPU'dadır.
> Detay: [Adım 3](#adım-3--bağımlılıkların-kurulumu-pytorch-dikkat) — CPU-only kurulum.

### 1.2 Yazılım

| Yazılım | Gerekli Sürüm | Doğrulama Kaynağı |
|---|---|---|
| **Windows** | 10 / 11 (x64) | Başlatma script'leri `.bat`; `worker/main.py` `COMPUTERNAME` fallback'i içerir. |
| **Python** | **3.11** (test edilen: 3.11.9) | `venv/pyvenv.cfg` → `version = 3.11.9`; `README.md` → "Python 3.11" |
| **PostgreSQL** | **13+** (önerilen 15/16/17) | `FOR UPDATE SKIP LOCKED` (9.5+), `pg_try_advisory_lock`, `JSONB`, `ARRAY`; `tests/conftest.py` içindeki `gen_random_uuid()` PG 13+ ile yerleşik gelir. |
| **Qdrant** | 1.18.x (client `qdrant-client==1.18.0` ile uyumlu) | `requirements.txt` |
| **Git** | Herhangi bir güncel sürüm | Repository klonlama |
| **LM Studio** *veya* **AWS hesabı** | — | `AI_PROVIDER` seçimine göre; bkz. [Adım 7](#adım-7--vlm-sağlayıcısı-lm-studio-veya-aws-bedrock) |

> **Docker kullanılmıyor.** Bu bilinçli bir karardır (`app/core/settings.py`,
> `start-photoai.bat` yorumları): backend, worker'lar ve Qdrant host üzerinde
> ayrı Windows süreçleri olarak çalışır.

---

## 2. Kullanılan Teknolojiler

`requirements.txt` **123 sabitlenmiş paket** içerir (`pip freeze` çıktısı). Öne çıkanlar:

### Web / API katmanı
| Paket | Sürüm | Rol |
|---|---|---|
| `fastapi` | 0.140.13 | HTTP API çatısı |
| `uvicorn` | 0.51.0 | ASGI sunucusu |
| `starlette` | 1.3.1 | FastAPI altyapısı, CORS middleware |
| `pydantic` / `pydantic-settings` | 2.13.4 / 2.15.0 | Şema doğrulama + `.env` okuma (`app/core/settings.py`) |
| `python-multipart` | 0.0.32 | `multipart/form-data` fotoğraf yükleme |
| `httpx` | 0.28.1 | LM Studio HTTP çağrıları + `/health` VLM kontrolü |

### Veritabanı / kuyruk
| Paket | Sürüm | Rol |
|---|---|---|
| `SQLAlchemy` | 2.0.51 | ORM (`app/db/models.py`), engine havuzu (`pool_size=10, max_overflow=15`) |
| `alembic` | 1.18.5 | Şema migration'ları (21 revision) |
| `psycopg2-binary` | 2.9.12 | PostgreSQL sürücüsü |
| `qdrant-client` | 1.18.0 | Vektör arama istemcisi |

### Kimlik doğrulama
| Paket | Sürüm | Rol |
|---|---|---|
| `PyJWT` | 2.13.0 | JWT access token |
| `bcrypt` | 5.0.0 | Parola hash'leme |
| `email-validator` | 2.3.0 | `EmailStr` doğrulaması (`scripts/create_admin.py`, `/auth/login`) |

### AI / ML
| Paket | Sürüm | Rol |
|---|---|---|
| `opencv-python` | 5.0.0.93 | YuNet yüz tespiti (`cv2.FaceDetectorYN`) |
| `onnxruntime` | 1.28.0 | ONNX çıkarım motoru (**CPU** sürümü) |
| `insightface` | 1.0.1 | AuraFace model yükleme + 5-nokta yüz hizalama (`face_align.norm_crop`) |
| `sentence-transformers` | 6.0.0 | Yerel e5 metin embedding (semantik arama) |
| `torch` | **2.5.1+cu121** | sentence-transformers backend'i — CUDA 12.1 tekerleği |
| `transformers` / `tokenizers` | 5.16.1 / 0.23.1 | e5 tokenizer/model yükleme |
| `hdbscan` | 0.8.44 | Yüz kümeleme (`app/services/clustering_service.py`) |
| `scikit-learn` / `scipy` / `numpy` | 1.9.0 / 1.17.1 / 2.3.5 | Kümeleme + vektör işlemleri |
| `boto3` / `botocore` | 1.43.72 | AWS Bedrock (VLM + Titan embedding) ve `/health` STS kontrolü |
| `pillow` / `pillow_heif` | 12.3.0 / 1.5.0 | Görsel işleme + **HEIC** desteği |

### Test
| Paket | Sürüm | Rol |
|---|---|---|
| `pytest` | 9.1.1 | 23 test dosyası (`tests/`) — **gerçek PostgreSQL'e karşı** çalışır |

---

## 3. Mimari Özet — Neyi Neden Kuruyoruz

Kurulumun neden bu kadar parçadan oluştuğunu anlamak, hataları teşhis etmeyi kolaylaştırır:

```
                      ┌───────────────────────────────────────────┐
   POST /photos ─────►│  BACKEND (uvicorn, port 8001)             │
   (202 döner)        │  ├─ FastAPI router'ları                   │
                      │  └─ ingestion döngüsü (arka plan thread'i) │
                      └────────┬──────────────────────────────────┘
                               │ tek atomik commit
                               ▼
   uploads/inbox/  ──►  photos + photo_exif + jobs  ──►  uploads/stored/
                               │
                               ▼  (PostgreSQL iş kuyruğu, FOR UPDATE SKIP LOCKED)
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  worker-face            worker-vlm            worker-semantic
  JOB_TYPES=             JOB_TYPES=            JOB_TYPES=
  face_pipeline          vlm_analysis          semantic_index
        │                      │                      │
        ▼                      ▼                      ▼
  YuNet + AuraFace       LM Studio / Bedrock    e5-base embedding
  (CPU, ONNX)            (VLM JSON analizi)     (GPU/CPU)
        │                      │                      │
        ▼                      ▼                      ▼
  Qdrant: faces,         PostgreSQL:            Qdrant:
  identity_pool          photo_analysis         photo_semantic
  (512-d)                                       (768-d)
```

**Kritik nokta:** Fotoğraf yükleme isteği **hiçbir AI işlemi yapmaz**, anında `202`
döner. Worker süreçleri çalışmıyorsa yüklenen fotoğraflar `queued` durumunda
sonsuza kadar bekler. Bu yüzden **üç worker'ın da** başlatılması gerekir.

---

## Adım 1 — Ön Koşul Yazılımların Kurulumu

### 1.1 Python 3.11

İndirme: **https://www.python.org/downloads/release/python-3119/**
(sayfanın altındaki *Windows installer (64-bit)*)

Kurulum sırasında **"Add python.exe to PATH"** kutusunu işaretleyin.

```powershell
python --version
```
**Beklenen çıktı:**
```
Python 3.11.9
```

> ⚠️ **3.12 / 3.13 kullanmayın.** `requirements.txt` içindeki `torch==2.5.1+cu121`
> tekerleği `cp311` içindir; `hdbscan` ve `insightface` de 3.11 üzerinde doğrulanmıştır.

### 1.2 Git

İndirme: **https://git-scm.com/download/win**

```powershell
git --version
```

### 1.3 PostgreSQL 13+

Resmî yönlendirme: **https://www.postgresql.org/download/windows/**
İndirme (EDB Interactive Installer): **https://www.enterprisedb.com/downloads/postgres-postgresql-downloads**

Kurulum sırasında:
- `postgres` süper kullanıcısı için bir parola belirleyin (`.env`'de kullanacaksınız).
- Port: **5432** (varsayılan).
- Locale: varsayılan.

Kurulum sonrası PATH'e ekleyin (pgAdmin yerine komut satırı kullanacaksanız):
```powershell
$env:Path += ";C:\Program Files\PostgreSQL\17\bin"
psql --version
```
**Beklenen çıktı:**
```
psql (PostgreSQL) 17.x
```

### 1.4 (Opsiyonel) NVIDIA sürücüsü + CUDA uyumluluğu

`torch==2.5.1+cu121` kendi CUDA runtime kütüphanelerini tekerlek içinde getirir —
**ayrı CUDA Toolkit kurmanıza gerek yoktur**. Sadece güncel bir NVIDIA sürücüsü
gerekir (CUDA 12.1+ destekleyen sürüm).

Sürücü: **https://www.nvidia.com/en-us/drivers/**

```powershell
nvidia-smi
```
**Beklenen çıktı (örnek):** GPU adı, sürücü sürümü ve "CUDA Version: 12.x" satırı.

---

## Adım 2 — Repository ve Python Sanal Ortamı

```powershell
cd C:\Users\<kullanici>\Desktop
git clone <repository-url> photoai-backend
cd C:\Users\<kullanici>\Desktop\photoai-backend
```

> **Klasör yerleşimi önemlidir.** `start-photoai.bat` (bkz.
> [Tüm Servisleri Tek Komutla Başlatma](#tüm-servisleri-tek-komutla-başlatma))
> kendi bulunduğu klasörün altında `photoai-backend\` ve `photoai-frontend\`
> klasörlerini arar:
> ```
> C:\Users\<kullanici>\Desktop\
> ├── start-photoai.bat
> ├── photoai-backend\      ← bu repository
> └── photoai-frontend\     ← ayrı repository (frontend, opsiyonel)
> ```

Sanal ortam oluşturma ve etkinleştirme:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Beklenen çıktı:** prompt'un başına `(venv)` gelir:
```
(venv) PS C:\Users\<kullanici>\Desktop\photoai-backend>
```

> **PowerShell "script çalıştırma devre dışı" hatası alırsanız:**
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> veya `cmd.exe` kullanın: `venv\Scripts\activate.bat`

pip'i güncelleyin:
```powershell
python -m pip install --upgrade pip
```

---

## Adım 3 — Bağımlılıkların Kurulumu (PyTorch dikkat!)

### 🔴 Kritik: `torch==2.5.1+cu121` PyPI'da YOKTUR

`requirements.txt` dosyasında PyTorch şu şekilde sabitlenmiştir:

```
torch==2.5.1+cu121
```

`+cu121` yerel sürüm etiketi (local version identifier) **yalnızca PyTorch'un kendi
indeks sunucusunda** bulunur. Düz `pip install -r requirements.txt` komutu
`ERROR: No matching distribution found for torch==2.5.1+cu121` hatasıyla başarısız olur.

### 3.1 GPU'lu kurulum (önerilen)

**Önce PyTorch'u kendi indeksinden kurun:**

```powershell
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Beklenen çıktı (son satır):**
```
Successfully installed torch-2.5.1+cu121 ...
```

**Sonra kalan bağımlılıkları kurun:**

```powershell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

> Resmî komut referansı: **https://pytorch.org/get-started/previous-versions/**
> (`pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121`)
> — bu proje `torchvision`/`torchaudio` kullanmadığı için yalnızca `torch` yeterlidir.

Bu adım ~3–6 GB indirir; bağlantınıza göre 10–30 dakika sürebilir.

### 3.2 CPU-only kurulum (GPU yoksa)

```powershell
pip install torch==2.5.1
pip install -r requirements.txt --no-deps --no-build-isolation 2>$null
pip install -r requirements.txt
```

Basitçe: önce PyPI'daki CPU `torch==2.5.1`'i kurun, ardından `requirements.txt`
içindeki `torch==2.5.1+cu121` satırını yerel bir kopyada `torch==2.5.1` olarak
düzeltip kurun:

```powershell
(Get-Content requirements.txt) -replace 'torch==2\.5\.1\+cu121', 'torch==2.5.1' | Set-Content requirements-cpu.txt -Encoding utf8
pip install -r requirements-cpu.txt
```

Ve `.env` dosyanızda mutlaka:
```
EMBEDDING_DEVICE=cpu
AI_PROVIDER=bedrock     # yerel VLM için GPU gerekir
```

> `requirements-cpu.txt` `.gitignore`'da değildir — commit etmeyin, yerel tutun.

### 3.3 Kurulumun doğrulanması

```powershell
python -c "import fastapi, sqlalchemy, cv2, onnxruntime, insightface, torch, qdrant_client; print('OK'); print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
```

**Beklenen çıktı (GPU'lu):**
```
OK
torch: 2.5.1+cu121 | cuda: True
```

**Beklenen çıktı (CPU-only):**
```
OK
torch: 2.5.1 | cuda: False
```

> `cuda: False` görüyor ama GPU'nuz varsa: PyTorch'un CPU sürümü kurulmuştur.
> `pip uninstall torch` yapıp [3.1](#31-gpulu-kurulum-önerilen) adımını tekrarlayın.

---

## Adım 4 — PostgreSQL Veritabanının Hazırlanması

### 4.1 Veritabanını oluşturun

```powershell
psql -U postgres -c "CREATE DATABASE photoai_db;"
```
**Beklenen çıktı:**
```
CREATE DATABASE
```

Bağlantıyı doğrulayın:
```powershell
psql -U postgres -d photoai_db -c "SELECT version();"
```
**Beklenen çıktı:** `PostgreSQL 17.x on x86_64-windows ...` benzeri tek satır.

### 4.2 (Opsiyonel) Test veritabanı

`tests/conftest.py` test veritabanını **otomatik oluşturur** —
üretim veritabanı adına `_test` soneki ekler (`photoai_db` → `photoai_db_test`)
ve migration'ları oraya uygular. Manuel bir işlem gerekmez.

### 4.3 Neden ek eklenti (extension) gerekmiyor?

Migration'ların hiçbiri `CREATE EXTENSION` çağırmaz. UUID değerleri Python
tarafında üretilir (`app/db/models.py` → `default=uuid.uuid4`). Yalnızca
`tests/conftest.py` içindeki geçici (TEMP) `jobs` tablosu `gen_random_uuid()`
kullanır — bu fonksiyon **PostgreSQL 13+ ile yerleşik** gelir, `pgcrypto`
kurmanıza gerek yoktur.

---

## Adım 5 — Qdrant Vektör Veritabanı

Qdrant, yüz embedding'lerinin (512-d) ve fotoğraf semantik vektörlerinin (768-d)
arama indeksidir. **PostgreSQL kaynak-of-truth'tur; Qdrant yalnızca türev veridir**
(`app/db/qdrant.py` başlığı).

### 5.1 İndirme

Resmî sürümler: **https://github.com/qdrant/qdrant/releases**

Windows için asset adı: `qdrant-x86_64-pc-windows-msvc.zip`

> `qdrant-client==1.18.0` ile eşleşmesi için **sunucu tarafında da 1.18.x**
> sürümünü tercih edin. (Yazım anındaki en güncel sürüm 1.19.1'dir; client 1.18
> ile genellikle uyumludur, ancak birebir eşleştirmek en güvenlisidir.)
> Resmî kurulum dokümantasyonu: **https://qdrant.tech/documentation/guides/installation/**

### 5.2 Kurulum

```powershell
New-Item -ItemType Directory -Force C:\qdrant
# İndirdiğiniz zip'i C:\qdrant altına açın; içinde qdrant.exe olmalı.
Get-ChildItem C:\qdrant
```

### 5.3 Çalıştırma

> 🔴 **`qdrant.exe` MUTLAKA kendi klasöründen çalıştırılmalıdır.** Qdrant, veri
> deposunu **çalışma dizinindeki** `.\storage` altında tutar. Başka bir dizinden
> başlatılırsa **boş bir depo** açar ve tüm yüz vektörleri kaybolmuş görünür
> (`start-photoai.bat` içindeki uyarı).

```powershell
cd C:\qdrant
.\qdrant.exe
```

**Beklenen çıktı (kısaltılmış):**
```
           _                 _
  __ _  __| |_ __ __ _ _ __ | |_
 / _` |/ _` | '__/ _` | '_ \| __|
...
Access web UI at http://localhost:6333/dashboard
```

Bu pencereyi **açık bırakın**. Yeni bir PowerShell penceresinden doğrulayın:

```powershell
curl.exe http://127.0.0.1:6333/collections
```
**Beklenen çıktı (ilk kurulumda koleksiyonlar henüz yok):**
```json
{"result":{"collections":[]},"status":"ok","time":0.000...}
```

> 🔴 **`localhost` DEĞİL, `127.0.0.1` kullanın.** `app/core/settings.py`'deki
> ölçüm notu: Windows'ta `localhost` önce IPv6'ya (`::1`) çözülüyor, Qdrant ise
> IPv4 dinliyor; başarısız IPv6 denemesi **her çağrıya ~2 sn** ekliyordu.

### 5.4 Koleksiyonlar

Koleksiyonları elle oluşturmanıza **gerek yoktur**. Backend açılışta
(`app/main.py` lifespan → `ensure_collections()`) şunları oluşturur:

| Koleksiyon | Boyut | Mesafe | İçerik |
|---|---|---|---|
| `faces` | 512 | Cosine | Tekil yüz embedding'leri |
| `identity_pool` | 512 | Cosine | Kişi (`person`) ve küme (`cluster`) merkezleri |
| `photo_semantic` | `EMBEDDING_DIM` (varsayılan **768**) | Cosine | Fotoğraf başına tek metin embedding'i |

---

## Adım 6 — AI/ML Model Dosyalarının İndirilmesi

`models/` klasörü **`.gitignore`'dadır** (ONNX ağırlıkları yüzlerce MB — repoyu
şişirir). Model dosyalarını elle indirmeniz gerekir.

Toplam indirilecek: **~1,4 GB**

```powershell
cd C:\Users\<kullanici>\Desktop\photoai-backend
New-Item -ItemType Directory -Force models\auraface
```

### 6.1 YuNet — Yüz Tespiti (~233 KB)

| | |
|---|---|
| **Hedef yol** | `models/face_detection_yunet_2023mar.onnx` |
| **Ayar** | `YUNET_MODEL_PATH` |
| **Kaynak** | OpenCV Zoo — https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| **Lisans** | MIT |
| **Kullanan kod** | `app/ai/face_detector.py` (`cv2.FaceDetectorYN`) |

```powershell
curl.exe -L -o models\face_detection_yunet_2023mar.onnx `
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

**Doğrulama:**
```powershell
(Get-Item models\face_detection_yunet_2023mar.onnx).Length
```
**Beklenen çıktı:** `232589` (yaklaşık; ~233 KB)

> ⚠️ Dosya boyutu **1 KB civarındaysa** Git LFS işaretçi metnini indirmişsinizdir.
> `-L` (redirect takibi) bayrağını kullandığınızdan emin olun.

### 6.2 AuraFace — Yüz Embedding (~261 MB)

| | |
|---|---|
| **Hedef yol** | `models/auraface/glintr100.onnx` |
| **Ayar** | `AURAFACE_MODEL_DIR` |
| **Kaynak** | Hugging Face — https://huggingface.co/fal/AuraFace-v1 |
| **Lisans** | Apache-2.0 (ticari kullanıma uygun — bu yüzden ArcFace yerine seçildi) |
| **Kullanan kod** | `app/ai/face_embedder.py` (`insightface.model_zoo`) |

`requirements.txt` ile birlikte gelen `huggingface_hub` CLI'ı (`hf`) ile:

```powershell
hf download fal/AuraFace-v1 glintr100.onnx --local-dir models\auraface
```

Alternatif (Python içinden — `face_embedder.py`'nin hata mesajındaki yöntem):

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('fal/AuraFace-v1', local_dir='models/auraface')"
```

Alternatif (doğrudan indirme):

```powershell
curl.exe -L -o models\auraface\glintr100.onnx `
  "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx"
```

**Doğrulama:**
```powershell
(Get-Item models\auraface\glintr100.onnx).Length
```
**Beklenen çıktı:** `260694151` (yaklaşık; ~249 MiB)

> Depoda `1k3d68.onnx`, `2d106det.onnx`, `genderage.onnx`, `scrfd_10g_bnkps.onnx`
> dosyaları da vardır. **Yalnızca `glintr100.onnx` gereklidir** — AuraFace'in
> kendi SCRFD dedektörü kullanılmıyor, yüz tespiti YuNet ile yapılıyor
> (`face_embedder.py` başlığı).

### 6.3 multilingual-e5-base — Semantik Arama Embedding'i (~1,13 GB)

| | |
|---|---|
| **Hedef yol** | `models/multilingual-e5-base/` (klasörün tamamı) |
| **Ayar** | `EMBEDDING_MODEL_DIR` |
| **Kaynak** | Hugging Face — https://huggingface.co/intfloat/multilingual-e5-base |
| **Lisans** | MIT |
| **Boyut (vektör)** | **768** — `EMBEDDING_DIM` ile eşleşmeli |
| **Kullanan kod** | `app/ai/embedding/local_e5.py` (`SentenceTransformer`) |

```powershell
hf download intfloat/multilingual-e5-base --local-dir models\multilingual-e5-base
```

**Doğrulama — sentence-transformers'ın ihtiyaç duyduğu dosyalar:**
```powershell
Get-ChildItem models\multilingual-e5-base | Select-Object Name, Length
```
**Beklenen çıktı (en az bu dosyalar bulunmalı):**
```
Name                        Length
----                        ------
1_Pooling                        (klasör)
config.json                         694
model.safetensors           1112201288
modules.json                        387
sentence_bert_config.json            57
sentencepiece.bpe.model         5069051
special_tokens_map.json             280
tokenizer.json                 17082660
tokenizer_config.json               418
```

> `SEMANTIC_SEARCH_ENABLED=false` yaparsanız bu modeli indirmeniz gerekmez;
> ancak semantik arama devre dışı kalır ve `worker-semantic` başlatmanıza gerek olmaz.

### 6.4 Model klasörünün son hâli

```
models/
├── face_detection_yunet_2023mar.onnx      ~233 KB    (YuNet, MIT)
├── auraface/
│   └── glintr100.onnx                     ~249 MiB   (AuraFace, Apache-2.0)
└── multilingual-e5-base/                  ~1,08 GiB  (intfloat e5, MIT)
    ├── config.json
    ├── model.safetensors
    ├── modules.json
    ├── sentence_bert_config.json
    ├── sentencepiece.bpe.model
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── 1_Pooling/config.json
```

---

## Adım 7 — VLM Sağlayıcısı (LM Studio veya AWS Bedrock)

VLM (Vision-Language Model), fotoğrafın içeriğini Türkçe JSON olarak analiz eder:
açıklama, ana nesne, ortam, aktivite, ruh hâli, etiketler, tanınan kamusal figürler
(`app/ai/dispatcher.py` → `PROMPT`).

İki seçenekten **birini** kurmanız gerekir.

### Seçenek A — LM Studio (yerel, GPU gerektirir)

| | |
|---|---|
| **Ayar** | `AI_PROVIDER=lm_studio` |
| **İndirme** | https://lmstudio.ai/ |
| **API dokümantasyonu** | https://lmstudio.ai/docs/app/api/endpoints/openai |
| **Varsayılan taban URL** | `http://localhost:1234/v1` |
| **Repository'de kullanılan model** | `qwen/qwen2.5-vl-7b` (`.env.example`) |
| **VRAM** | ~5,6 GB (ölçülmüş) |

**Kurulum:**
1. LM Studio'yu kurun ve açın.
2. **Discover** sekmesinden `Qwen2.5-VL-7B` (vision destekli) modelini indirin.
3. **Developer / Local Server** sekmesinden modeli yükleyip sunucuyu başlatın.
   Varsayılan port: **1234**.

**Doğrulama:**
```powershell
curl.exe http://localhost:1234/v1/models
```
**Beklenen çıktı:** yüklü modelleri listeleyen bir JSON (`"object": "list"`, `"data": [...]`).
`dispatcher.py` ve `/health` bu ucu kullanır.

> LM Studio kullanıyorsanız `VLM_BASE_URL` **`/v1` ile bitmelidir**
> (`dispatcher._chat_url()` gerekirse ekler ama açıkça yazmak nettir).

### Seçenek B — AWS Bedrock (bulut, GPU gerektirmez)

| | |
|---|---|
| **Ayar** | `AI_PROVIDER=bedrock` |
| **Dokümantasyon** | https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html |
| **Kimlik bilgisi kurulumu** | https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html |
| **Model erişimi** | https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html |
| **API** | Converse API (boto3 `bedrock-runtime`) |

> 🔒 **AWS kimlik bilgileri `.env`'de TUTULMAZ.** Kod, boto3'un standart kimlik
> bilgisi zincirini kullanır (`app/core/settings.py` yorumu): ortam değişkenleri,
> `~/.aws/credentials` veya IAM rolü.

**Kurulum:**
1. AWS CLI kurun: https://aws.amazon.com/cli/
2. Kimlik bilgilerini yapılandırın:
   ```powershell
   aws configure
   ```
   (Access Key ID, Secret Access Key, `us-east-1`, `json`)
3. AWS konsolunda **Bedrock → Model access** üzerinden kullanacağınız modele
   erişim talep edin.
4. `.env` içinde `AWS_BEDROCK_MODEL_ID` değerini seçtiğiniz modelin kimliğiyle
   doldurun. `.env.example` varsayılanı:
   `anthropic.claude-3-5-sonnet-20241022-v2:0`

**Doğrulama:**
```powershell
aws sts get-caller-identity
```
**Beklenen çıktı:** `UserId`, `Account`, `Arn` alanlarını içeren JSON.

> `/health` ucu Bedrock modunda **modeli çağırmaz** (ücretli + yavaş); yalnızca
> `sts:GetCallerIdentity` ile kimliğin geçerli olduğunu doğrular (`app/main.py`
> `_check_vlm()`).

### Semantik arama için Bedrock alternatifi

`EMBEDDING_PROVIDER=bedrock` ile yerel e5 yerine **Titan Text Embeddings V2**
(`amazon.titan-embed-text-v2:0`) kullanılabilir. Bu durumda:
- `EMBEDDING_DIM` **256, 512 veya 1024** olmalıdır (`bedrock_titan.py` doğrular),
- Qdrant `photo_semantic` koleksiyonu düşürülüp yeniden backfill edilmelidir,
- `models/multilingual-e5-base` indirmeye gerek kalmaz.

---

## Adım 8 — `.env` Dosyasının Oluşturulması

```powershell
copy .env.example .env
```

### 8.1 Zorunlu değerler (varsayılanı YOK — eksikse uygulama başlamaz)

`app/core/settings.py` içinde varsayılansız tanımlı dört alan:

| Değişken | Açıklama |
|---|---|
| `DATABASE_URL` | PostgreSQL bağlantı adresi |
| `VLM_BASE_URL` | **`AI_PROVIDER=bedrock` olsa bile zorunludur** (alanın varsayılanı yok) |
| `VLM_MODEL` | **`AI_PROVIDER=bedrock` olsa bile zorunludur** |
| `JWT_SECRET` | Token imzalama anahtarı |

`JWT_SECRET` üretmek için (`.env.example`'daki komut):

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 8.2 Örnek `.env` — yerel LM Studio + GPU

```ini
# --- VERİTABANI ---
DATABASE_URL=postgresql://postgres:ORNEK_DB_PAROLASI@localhost:5432/photoai_db

# --- VLM ---
AI_PROVIDER=lm_studio
VLM_BASE_URL=http://localhost:1234/v1
VLM_MODEL=qwen/qwen2.5-vl-7b

# --- KİMLİK DOĞRULAMA ---
JWT_SECRET=ORNEK_UZUN_RASTGELE_DEGER_BURAYA
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=30

# --- CORS (frontend adresi) ---
CORS_ORIGINS=http://localhost:5173

# --- YÜZ TANIMA MODELLERİ ---
YUNET_MODEL_PATH=models/face_detection_yunet_2023mar.onnx
AURAFACE_MODEL_DIR=models/auraface

# --- QDRANT ---
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=

# --- SEMANTİK ARAMA ---
SEMANTIC_SEARCH_ENABLED=true
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_DIR=models/multilingual-e5-base
EMBEDDING_DEVICE=cuda
EMBEDDING_DIM=768
```

### 8.3 Örnek `.env` — AWS Bedrock + CPU (GPU'suz makine)

```ini
DATABASE_URL=postgresql://postgres:ORNEK_DB_PAROLASI@localhost:5432/photoai_db

# VLM_BASE_URL / VLM_MODEL bedrock modunda KULLANILMAZ ama ZORUNLUDUR
AI_PROVIDER=bedrock
VLM_BASE_URL=http://localhost:1234/v1
VLM_MODEL=unused-in-bedrock-mode
AWS_REGION=us-east-1
AWS_BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
# AWS anahtarları BURADA DEĞİL — `aws configure` ile ~/.aws/credentials içinde

JWT_SECRET=ORNEK_UZUN_RASTGELE_DEGER_BURAYA
CORS_ORIGINS=http://localhost:5173

QDRANT_URL=http://127.0.0.1:6333

SEMANTIC_SEARCH_ENABLED=true
EMBEDDING_PROVIDER=local
EMBEDDING_DEVICE=cpu
EMBEDDING_DIM=768
```

> 🔒 `.env` **`.gitignore`'dadır** — asla commit etmeyin. Yukarıdaki tüm parola/
> anahtar değerleri **örnektir**, gerçek değerlerinizi kullanın.

### 8.4 CORS notu

`allow_credentials=True` olduğu için **`*` joker karakteri kullanılamaz**
(`app/main.py`). Frontend başka bir makinede/cihazda açılıyorsa tam adresi
virgülle ekleyin:

```ini
CORS_ORIGINS=http://192.168.1.50:5173,http://localhost:5173
```

---

## Adım 9 — Alembic Migration'ları

Repository'de **21 migration** vardır. Zincirin başı `921dfba930d5`,
**head'i `b4d7e2a9c153`** (`content_hash_unique`).

`alembic/env.py` veritabanı adresini `alembic.ini`'den değil,
**`settings.DATABASE_URL`'den (yani `.env`'den) okur** — `alembic.ini` içine
elle URL yazmayın.

```powershell
alembic upgrade head
```

**Beklenen çıktı (kısaltılmış — 21 satır sürer):**
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 921dfba930d5, photos ve photo_analysis tablolari
INFO  [alembic.runtime.migration] Running upgrade 921dfba930d5 -> b05ab8dd4e2f, photo_analysis zengin alanlar
...
INFO  [alembic.runtime.migration] Running upgrade a2c4e6f8b0d1 -> b4d7e2a9c153, content_hash unique
```

**Doğrulama:**
```powershell
alembic current
```
**Beklenen çıktı:**
```
b4d7e2a9c153 (head)
```

Tabloların oluştuğunu görmek için:
```powershell
psql -U postgres -d photoai_db -c "\dt"
```
**Beklenen çıktı:** `users`, `refresh_tokens`, `photos`, `photo_analysis`,
`photo_exif`, `photo_embeddings`, `faces`, `persons`, `clusters`,
`cluster_constraints`, `albums`, `jobs`, `user_job_counters`,
`worker_heartbeats`, `activity_log`, `alembic_version` tabloları.

### Migration zinciri (kronolojik)

| # | Revision | Konu |
|---|---|---|
| 1 | `921dfba930d5` | photos ve photo_analysis tabloları |
| 2 | `b05ab8dd4e2f` | photo_analysis zengin alanlar |
| 3 | `d3f4a1b2c9e7` | yüz tanıma pipeline tabloları |
| 4 | `e7c1b8a4f2d9` | faces.is_background |
| 5 | `a4b1c9d2e5f6` | photos.content_hash |
| 6 | `b7c2e8f4a1d3` | face_suggestions kaldırıldı |
| 7 | `c1e9a2f6b3d4` | çoklu kullanıcı auth |
| 8 | `d5a3f1c7b9e2` | jobs kuyruğu |
| 9 | `e8b4d2f1a6c3` | jobs.photo_id index |
| 10 | `f1a2b3c4d5e6` | jobs.failure_count |
| 11 | `a7b3c9d1e5f2` | persons/clusters centroid |
| 12 | `b1c4d6e8f0a2` | user delete FK ondelete |
| 13 | `c2d5e7f9a1b3` | albümler |
| 14 | `a1f5d8c2b4e6` | photo_analysis alan sadeleştirme |
| 15 | `b2e6c9a4f1d7` | cluster_constraints.source |
| 16 | `c3f7a2e8b5d1` | activity_log |
| 17 | `d4a9c6b2e3f8` | photo_exif |
| 18 | `e1f4a7b2c5d8` | system service user |
| 19 | `f2b5c8d1e4a7` | photo_embeddings |
| 20 | `a2c4e6f8b0d1` | worker_heartbeats |
| 21 | `b4d7e2a9c153` | **content_hash unique (HEAD)** |

---

## Adım 10 — İlk Admin Hesabı

Sistem **açık self-register içermez** — yeni hesapları yalnızca `admin` rolündeki
bir kullanıcı `POST /users` ile oluşturabilir. İlk admin hesabı bir kereye mahsus
script ile kurulur.

```powershell
python scripts\create_admin.py --email admin@ornek.com --password ORNEK_GUCLU_PAROLA --display-name "Ad Soyad"
```

**Beklenen çıktı:**
```
Admin kullanici olusturuldu: admin@ornek.com (id=3f2a1c9e-...-...)
```

**Script'in doğrulamaları (`scripts/create_admin.py`):**
- E-posta `EmailStr` ile doğrulanır → `.local` / `.test` gibi rezerve TLD'ler reddedilir
  (aksi hâlde oluşan hesapla API'ye giriş yapılamazdı).
- Parola **en az 8 karakter** olmalıdır.
- Aynı e-posta varsa hata verir, **parolayı sıfırlamaz**.

**Roller:**

| Rol | Yetki |
|---|---|
| `admin` | Kullanıcı yönetimi + tüm işlemler |
| `editor` | Yükleme, etiketleme, birleştirme, silme |
| `viewer` | Salt okunur |

---

## Adım 11 — Backend'i Başlatma

### Geliştirme

```powershell
uvicorn app.main:app --reload --reload-dir app --port 8001
```

**Beklenen çıktı:**
```
INFO:     Will watch for changes in these directories: ['C:\\...\\photoai-backend\\app']
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
2026-09-06 21:00:00,123 [INFO] photoai.ingestion: [INGEST] dongu basladi (poll=1.0s reconcile=30s inbox=uploads\inbox)
INFO:     Application startup complete.
```

### Üretim

```powershell
uvicorn app.main:app --port 8001
```

> 🔴 **`--reload` üretimde kullanılmamalıdır** (`start-photoai.bat` uyarısı):
> yeniden yükleme sırasında kısa süre **iki backend süreci** (dolayısıyla iki
> ingestion döngüsü) var olur. Veri açısından güvenlidir — her dosya
> içerik-scoped advisory kilit altında işlenir — ama gereksiz kaynak tüketir ve
> logları ikizler. Ayrıca kod değişikliğinde süren yüklemeler kopar.

### Neden `--reload-dir app`?

Düz `--reload` proje kökünü izler; buranın %99'u `venv\` içindeki ~44.000
kütüphane dosyasıdır. Windows'ta bu kadar dosyayı sürekli taramak CPU yer ve
yüz tanıma çıkarımını yavaşlatır — **ölçüm: fotoğraf başına 1,97 sn → 0,29 sn
(~6 kat)** (`start-photoai.bat`).

### Neden port 8001?

Windows'ta port 8000 bazen "hayalet dinleyici" hâlinde takılı kalıyor
(`taskkill`/`Get-Process` ile temizlenemiyor, yalnızca yeniden başlatma çözüyor).
Backend bu yüzden **kalıcı olarak 8001'e** taşındı. Frontend'in `VITE_API_URL`
değeri bununla **eşleşmelidir**.

---

## Adım 12 — Worker Süreçlerini Başlatma

**Worker'lar olmadan yüklenen fotoğraflar sonsuza kadar `queued` durumunda bekler.**

Her worker **ayrı bir terminal penceresinde**, sanal ortam etkinleştirilmiş
hâlde çalıştırılmalıdır.

### PowerShell

```powershell
# Pencere 1 — VLM analizi
$env:JOB_TYPES="vlm_analysis"; python -m app.worker.main
```
```powershell
# Pencere 2 — Yüz hattı
$env:JOB_TYPES="face_pipeline"; python -m app.worker.main
```
```powershell
# Pencere 3 — Semantik indeksleme
$env:JOB_TYPES="semantic_index"; python -m app.worker.main
```

### cmd.exe (README ve `.bat` dosyasındaki söz dizimi)

```bat
set JOB_TYPES=vlm_analysis  && python -m app.worker.main
set JOB_TYPES=face_pipeline && python -m app.worker.main
set JOB_TYPES=semantic_index && python -m app.worker.main
```

**Beklenen çıktı (her pencerede):**
```
2026-09-06 21:00:05,001 [INFO] photoai.worker: Worker basladi: id=DESKTOP-ABC123-12345 tipler=vlm_analysis
```

Bir iş alındığında:
```
2026-09-06 21:00:12,456 [INFO] photoai.worker: Is alindi: job=8f2c... type=vlm_analysis deneme=1/3
```

### İş tipleri

| `JOB_TYPES` | Ne yapar | Kuyruğa kim yazar |
|---|---|---|
| `face_pipeline` | YuNet tespiti + AuraFace embedding + kimlik atama/kümeleme | `ingestion_service` (her yükleme) |
| `vlm_analysis` | VLM ile içerik analizi → `photo_analysis` | `ingestion_service` (her yükleme) |
| `semantic_index` | Analiz JSON'unu e5 ile embed edip Qdrant `photo_semantic`'e yazar | `photo_service` (VLM analizi **başarıyla** bittikten sonra) |

### 🔴 `JOB_TYPES` zorunludur — worker onsuz başlamaz

`app/worker/main.py` → `parse_job_types()` **fail-fast** davranışı uygular.
"Varsayılan olarak tümünü tüket" davranışı **bilinçli olarak yoktur**: yanlış
yapılandırılmış bir `worker-face` süreci sessizce VLM işlerini de çekerse tek
GPU'daki eşzamanlılık kontrolü çöker.

**Boş `JOB_TYPES` ile beklenen çıktı:**
```
HATA: JOB_TYPES tanimsiz ya da bos. Worker hangi is tipini tuketecegini bilmeden
BASLAMAZ (varsayilan 'tumunu tuket' davranisi bilincli olarak yoktur).
Gecerli tipler: face_pipeline, semantic_index, vlm_analysis
Ornek: JOB_TYPES=vlm_analysis
```

### 🔴 Süreç sayısı sınırları

| Ayar | Varsayılan | Neden artırılmamalı |
|---|---|---|
| `WORKER_VLM_PROCESSES` | **1** | Tek GPU paylaşılıyor; VLM modeli tek başına 5,6 GB / 6,1 GB VRAM kullanıyor — ikinci süreç VRAM'i taşırır. |
| `WORKER_FACE_PROCESSES` | **1** | Qdrant `identity_pool` centroid güncellemelerinde **lost update** riski: iki süreç aynı kimliğin centroid'ini aynı anda güncellerse biri diğerini ezer. 1'in üzerine çıkmadan önce `app/db/jobs_repository.py` başındaki not ile `identity_locks.py` / `locks.py` okunmalıdır. |

> Tek bir worker'a birden fazla tip vermek (`JOB_TYPES=vlm_analysis,face_pipeline`)
> **çalışır ama tavsiye edilmez**: `claim_next` yavaş `ANY` yoluna düşer
> (ölçüm: ~266 kat yavaş) ve worker bunu bir `WARNING` ile bildirir.

### Graceful shutdown

`Ctrl+C` (SIGINT) worker'a **mevcut işi bitir, yeni iş alma** der. Kapanışta
heartbeat kaydı silinir:
```
2026-09-06 21:05:00,000 [INFO] photoai.worker: Kapanma istegi alindi - mevcut is bitince duracak.
2026-09-06 21:05:03,000 [INFO] photoai.worker: Worker temiz sekilde durdu: id=DESKTOP-ABC123-12345
```

---

## Adım 13 — Kurulumun Doğrulanması

### 13.1 `/health` — bağımlılıkların tümü

```powershell
curl.exe http://localhost:8001/health
```

**Beklenen çıktı (her şey doğruysa):**
```json
{
  "status": "ok",
  "checks": {
    "database": { "status": "ok" },
    "qdrant":   { "status": "ok" },
    "vlm":      { "status": "ok" }
  },
  "workers": {
    "face_pipeline":  { "status": "ok", "workers": 1, "last_seen": "2026-09-06T21:00:05+00:00", "seconds_since": 3.2 },
    "vlm_analysis":   { "status": "ok", "workers": 1, "last_seen": "...", "seconds_since": 2.8 },
    "semantic_index": { "status": "ok", "workers": 1, "last_seen": "...", "seconds_since": 4.1 }
  },
  "ingestion": {
    "status": "ok",
    "inbox":  { "ready": 0, "in_progress": 0, "orphan_meta": 0, "failed": 0 },
    "totals": { "created": 0, "duplicate": 0, "recovered": 0, "failed": 0 },
    "last_reconcile_at": 1757188800.0
  }
}
```

**Alanların anlamı (`app/main.py`):**

| Alan | Anlam |
|---|---|
| `status` | Tüm `checks` **ve** tüm `workers` `ok` ise `ok`, aksi hâlde `degraded` |
| `checks.database` | `SELECT 1` |
| `checks.qdrant` | `get_collections()` |
| `checks.vlm` | `lm_studio` → `GET {VLM_BASE_URL}/models`; `bedrock` → `sts:GetCallerIdentity` |
| `workers.<tip>.status` | `ok` (~son 45 sn içinde heartbeat) / `stale` (kayıt var, heartbeat eskimiş) / `down` (o tip için hiç worker başlamamış) |
| `ingestion.inbox.ready` | İşlenmeyi bekleyen dosya sayısı — **sürekli büyüyorsa döngü takılmıştır** |
| `ingestion.inbox.orphan_meta` | `> 0` ise commit/taşıma arasında crash yaşanmış, uzlaştırma bekleniyor |

> `"status": "degraded"` + `workers.semantic_index.status: "down"` en sık görülen
> durumdur: üçüncü worker'ı başlatmayı unutmuşsunuzdur.

### 13.2 Swagger UI

Tarayıcıda: **http://localhost:8001/docs**

**Beklenen:** `auth`, `users`, `jobs`, `photos`, `faces`, `system`, `albums`
etiketleri altında uçların listelendiği Swagger arayüzü.

### 13.3 Giriş yapma

```powershell
curl.exe -X POST http://localhost:8001/auth/login `
  -H "Content-Type: application/json" `
  -d "{\"email\":\"admin@ornek.com\",\"password\":\"ORNEK_GUCLU_PAROLA\"}"
```

**Beklenen çıktı:**
```json
{"access_token":"eyJhbGciOiJIUzI1NiIs...","refresh_token":"...","token_type":"bearer"}
```

### 13.4 Uçtan uca test — fotoğraf yükleme

```powershell
$token = "<yukaridaki access_token>"
curl.exe -X POST http://localhost:8001/photos `
  -H "Authorization: Bearer $token" `
  -F "file=@C:\yol\ornek.jpg"
```

**Beklenen:** HTTP **202** (senkron AI işlemi yok — dosya inbox'a yazıldı).

Ardından kuyruğu izleyin:
```powershell
curl.exe -H "Authorization: Bearer $token" http://localhost:8001/jobs/queue-status
```

**Beklenen akış (birkaç saniye içinde):**
1. `face_pipeline` ve `vlm_analysis` işleri `queued` → `running` → `done`
2. `vlm_analysis` bittikten sonra bir `semantic_index` işi belirir ve o da `done` olur
3. `GET /photos` çağrısında fotoğraf analiz sonuçlarıyla birlikte döner

Worker pencerelerinde eş zamanlı log akışı görmelisiniz.

### 13.5 Qdrant koleksiyonları

```powershell
curl.exe http://127.0.0.1:6333/collections
```
**Beklenen çıktı (backend en az bir kez başladıktan sonra):**
```json
{"result":{"collections":[{"name":"faces"},{"name":"identity_pool"},{"name":"photo_semantic"}]},"status":"ok","time":0.000...}
```

### 13.6 Mevcut arşiv için semantik indeks (opsiyonel)

Sisteme daha önce yüklenmiş, VLM analizi olan ama semantik indeksi olmayan
fotoğraflar varsa:

```powershell
python scripts\backfill_photo_semantic.py
```

Seçenekler (`scripts/backfill_photo_semantic.py` başlığı):
```powershell
python scripts\backfill_photo_semantic.py --chunk-size 200
python scripts\backfill_photo_semantic.py --reindex-all
python scripts\backfill_photo_semantic.py --resume-after <uuid>
```

> Script **idempotenttir** — kesintiye uğrarsa baştan çalıştırmak güvenlidir.

---

## Adım 14 — Testlerin Çalıştırılması (opsiyonel)

```powershell
pytest
```

**Testler ÜRETİM veritabanını kullanmaz.** `tests/conftest.py`, `.env`'deki
`DATABASE_URL`'den `_test` sonekli bir veritabanı adı türetir
(`photoai_db` → `photoai_db_test`), yoksa **otomatik oluşturur** ve migration'ları
oraya uygular.

**Beklenen ilk çalıştırma çıktısı:**
```
[conftest] test veritabani olusturuldu: photoai_db_test
INFO  [alembic.runtime.migration] Running upgrade ...
============================= test session starts ==============================
...
```

Kaçış kapısı (paylaşımlı/üretim DB ile eski davranış):
```powershell
$env:PHOTOAI_TEST_USE_MAIN_DB="1"; pytest
```

> Testler **çalışan bir PostgreSQL** gerektirir. Ayrıca her test kendi geçici
> `uploads/` kökünü kullanır (autouse `isolated_storage_dirs` fixture'ı) — canlı
> ingestion döngüsü test dosyalarını "kapmaz".

---

## Tüm Servisleri Tek Komutla Başlatma

Repository'nin **bir üst klasöründe** bulunan `start-photoai.bat`, tüm servisleri
ayrı pencerelerde başlatır. Bu dosya **repository'nin parçası değildir**
(README `..\start-photoai.bat` olarak referans verir) — yoksa aşağıdaki içerikle
oluşturabilirsiniz:

```
C:\Users\<kullanici>\Desktop\
├── start-photoai.bat     ← burada
├── photoai-backend\
└── photoai-frontend\
```

Script'in yaptıkları (sırayla):

| Adım | Servis | Komut / Pencere adı |
|---|---|---|
| 1 | **Qdrant** | `C:\qdrant` içinde `qdrant.exe` — 6333 dinleniyorsa atlanır, 6 sn beklenir |
| 2 | **Backend** | `uvicorn app.main:app --reload --reload-dir app --port 8001` — *"PhotoAI Backend"* |
| 2b | *(Ingestion)* | **Ayrı pencere gerekmez** — backend süreci içinde arka plan thread'i |
| 3a | **worker-vlm** | `set JOB_TYPES=vlm_analysis && python -m app.worker.main` — *"PhotoAI Worker VLM"* |
| 3b | **worker-face** | `set JOB_TYPES=face_pipeline && python -m app.worker.main` — *"PhotoAI Worker Face"* |
| 3c | **worker-semantic** | `set JOB_TYPES=semantic_index && python -m app.worker.main` — *"PhotoAI Worker Semantic"* |
| 4 | **Frontend** | `npm run dev` (photoai-frontend) |

Başlatma sonrası adresler:

| Servis | Adres |
|---|---|
| Qdrant Dashboard | http://127.0.0.1:6333/dashboard |
| Backend (Swagger) | http://localhost:8001/docs |
| Frontend | http://localhost:5173 |

> Ingestion'ı ayrı bir sürece almak isterseniz: `.env`'de
> `INGESTION_ENABLED=false` yapıp `python -m app.ingestion.main` çalıştırın.

---

## Ortam Değişkenleri — Tam Referans

`app/core/settings.py` (pydantic-settings) — `.env` otomatik okunur,
tanımsız anahtarlar yok sayılır (`extra="ignore"`).

### Zorunlu (varsayılanı yok)

| Değişken | Örnek | Not |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:ORNEK@localhost:5432/photoai_db` | |
| `VLM_BASE_URL` | `http://localhost:1234/v1` | Bedrock modunda da zorunlu |
| `VLM_MODEL` | `qwen/qwen2.5-vl-7b` | Bedrock modunda da zorunlu |
| `JWT_SECRET` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` | |

### VLM sağlayıcı

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `AI_PROVIDER` | `lm_studio` | `lm_studio` \| `bedrock` |
| `AWS_REGION` | `us-east-1` | Bedrock + STS bölgesi |
| `AWS_BEDROCK_MODEL_ID` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | |

### Kimlik doğrulama

| Değişken | Varsayılan |
|---|---|
| `JWT_ALGORITHM` | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `30` |

### CORS

| Değişken | Varsayılan | Not |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173` | Virgülle ayrılmış; `*` **kullanılamaz** |

### Yüz tanıma & Qdrant

| Değişken | Varsayılan |
|---|---|
| `YUNET_MODEL_PATH` | `models/face_detection_yunet_2023mar.onnx` |
| `AURAFACE_MODEL_DIR` | `models/auraface` |
| `QDRANT_URL` | `http://127.0.0.1:6333` |
| `QDRANT_API_KEY` | *(boş)* |
| `IDENTITY_SEARCH_BACKEND` | `qdrant` | 🔴 `pg_brute_force` **açılamaz** — validator uygulama başlangıcında reddeder (PR-D tamamlanmadı) |

### Semantik arama

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `SEMANTIC_SEARCH_ENABLED` | `true` | |
| `EMBEDDING_PROVIDER` | `local` | `local` \| `bedrock` |
| `EMBEDDING_MODEL_DIR` | `models/multilingual-e5-base` | |
| `EMBEDDING_DEVICE` | `cuda` | `cuda` \| `cpu` (yalnızca `local`) |
| `EMBEDDING_DIM` | `768` | e5-base=768, Titan v2=256/512/1024. **Değişirse koleksiyon düşürülüp backfill edilmeli** |
| `AWS_BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | |
| `SEMANTIC_SEARCH_TOP_K` | `200` | |
| `SEMANTIC_SEARCH_MIN_SCORE` | `0.80` | e5-base kalibrasyonu: alakasız çiftler bile ~0.70–0.78 alır; ayırt edici aralık ~0.80–0.85. `0.0` = eşik kapalı |
| `SYSTEM_USER_ID` | `00000000-0000-0000-0000-000000000001` | Otomatik `semantic_index` işleri için servis hesabı (`is_active=False`) |

### İş kuyruğu / worker

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `JOB_TYPES` | *(boş)* | 🔴 Boşsa worker **başlamaz**. Ortam değişkeni `.env`'i ezer |
| `WORKER_CONCURRENCY` | `1` | |
| `JOB_HEARTBEAT_SECONDS` | `30` | Çalışan **işin** kilidini tazeler |
| `JOB_STALE_TIMEOUT_SECONDS` | `300` | Reaper eşiği |
| `JOB_POLL_INTERVAL_SECONDS` | `1.0` | |
| `JOB_REAP_INTERVAL_SECONDS` | `60` | |
| `WORKER_HEARTBEAT_INTERVAL_SECONDS` | `15` | Worker **sürecinin** canlılık yazımı |
| `WORKER_HEARTBEAT_STALE_SECONDS` | `45` | `/health`'in `stale` eşiği (3 kaçan vuruş) |
| `JOB_LOCK_CONFLICT_BACKOFF_SECONDS` | `20` | Advisory lock çakışmasında bekleme tabanı |
| `JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS` | `10` | Aşılınca **otomatik fail yok**, yalnızca `logger.error` ile eskalasyon |
| `JOB_ETA_SAMPLE_SIZE` | `20` | `queue-status` ETA örneklem boyutu |
| `WORKER_VLM_PROCESSES` | `1` | 🔴 VRAM sınırı |
| `WORKER_FACE_PROCESSES` | `1` | 🔴 Centroid lost-update riski |

### Ingestion (inbox)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `INGESTION_ENABLED` | `true` | `false` → backend içinde döngü başlamaz; `python -m app.ingestion.main` gerekir |
| `INGESTION_POLL_INTERVAL_SECONDS` | `1.0` | Yeni dosyayı yakalama gecikmesi |
| `INGESTION_RECONCILE_INTERVAL_SECONDS` | `30` | Tam uzlaştırma (açılışta bir kez de çalışır) |
| `INGESTION_STALE_PART_SECONDS` | `1800` | Bu süredir dokunulmamış `.part` → `failed/` |
| `INGESTION_BATCH_LIMIT` | `50` | Tek turda işlenecek en fazla dosya |

### Hibrit kimlik atama (deneysel)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `HYBRID_ASSIGNMENT_ENABLED` | `false` | Gerçek atamayı etkilemesi için **ikisi birden** gerekir: bu `true` **ve** `HYBRID_SHADOW_MODE=false` |
| `HYBRID_SHADOW_MODE` | `true` | Gölge modda kararlar yalnızca `logs/hybrid_assignment_decisions.jsonl`'a yazılır |
| `HYBRID_CONFIDENCE_THRESHOLD` | `0.55` | |
| `HYBRID_GAP_THRESHOLD` | `0.05` | |
| `HYBRID_TOP_CANDIDATES` | `5` | |
| `HYBRID_QUALITY_WEIGHTING_ENABLED` | `true` | |

### `.env` dışı ortam değişkenleri

| Değişken | Kullanan | Açıklama |
|---|---|---|
| `PHOTOAI_UPLOAD_ROOT` | `app/services/photo_service.py` | `uploads/` kökünü yönlendirir (testler, yan yana ikinci örnek) |
| `PHOTOAI_BACKUP_ROOT` | `scripts/backup_system.py` | Yedek klasörü; boşsa `Masaüstü/PhotoAI-Backups` |
| `PHOTOAI_TEST_USE_MAIN_DB` | `tests/conftest.py` | `1` → testler üretim DB'sini kullanır |
| `PHOTOAI_TEST_DB_NAME` | `tests/conftest.py` | Test DB adını elle belirler |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | boto3 | **`.env`'e YAZILMAZ** — `aws configure` veya IAM rolü |

---

## Kurulum Checklist'i

Sırayla işaretleyin:

### Ön koşullar
- [ ] Python **3.11.x** kurulu, PATH'te (`python --version`)
- [ ] Git kurulu
- [ ] PostgreSQL **13+** kurulu ve çalışıyor, `postgres` parolası biliniyor
- [ ] (GPU'lu kurulum için) NVIDIA sürücüsü güncel (`nvidia-smi`)
- [ ] Qdrant `C:\qdrant\qdrant.exe` olarak açıldı

### Proje
- [ ] Repository klonlandı (`photoai-backend`, tercihen `start-photoai.bat` ile aynı üst klasörde)
- [ ] `python -m venv venv` + `.\venv\Scripts\Activate.ps1` → prompt'ta `(venv)`
- [ ] **PyTorch önce kuruldu:** `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`
- [ ] `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`
- [ ] Doğrulama: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### Model dosyaları
- [ ] `models/face_detection_yunet_2023mar.onnx` var (~233 KB, 1 KB **değil**)
- [ ] `models/auraface/glintr100.onnx` var (~249 MiB)
- [ ] `models/multilingual-e5-base/` var (`config.json`, `model.safetensors`, `modules.json`, `1_Pooling/`, tokenizer dosyaları)

### Servisler
- [ ] `CREATE DATABASE photoai_db;` çalıştırıldı
- [ ] Qdrant çalışıyor: `curl http://127.0.0.1:6333/collections` → `{"result":{"collections":[]}...}`
- [ ] VLM sağlayıcısı hazır: LM Studio `curl http://localhost:1234/v1/models` **veya** `aws sts get-caller-identity`

### Yapılandırma
- [ ] `copy .env.example .env`
- [ ] `DATABASE_URL`, `VLM_BASE_URL`, `VLM_MODEL`, `JWT_SECRET` **dördü de** dolduruldu
- [ ] `JWT_SECRET` rastgele üretildi (örnek değer bırakılmadı)
- [ ] `CORS_ORIGINS` frontend adresini içeriyor
- [ ] GPU yoksa: `EMBEDDING_DEVICE=cpu` + `AI_PROVIDER=bedrock`

### Veritabanı ve hesap
- [ ] `alembic upgrade head` başarılı
- [ ] `alembic current` → `b4d7e2a9c153 (head)`
- [ ] `python scripts\create_admin.py --email ... --password ... --display-name ...` → "Admin kullanici olusturuldu"

### Çalıştırma
- [ ] Backend: `uvicorn app.main:app --reload --reload-dir app --port 8001` → "Application startup complete"
- [ ] worker-vlm penceresi: `Worker basladi: ... tipler=vlm_analysis`
- [ ] worker-face penceresi: `Worker basladi: ... tipler=face_pipeline`
- [ ] worker-semantic penceresi: `Worker basladi: ... tipler=semantic_index`

### Doğrulama
- [ ] `curl http://localhost:8001/health` → `"status": "ok"` (checks **ve** workers hepsi `ok`)
- [ ] http://localhost:8001/docs açılıyor
- [ ] `POST /auth/login` token döndürüyor
- [ ] Örnek fotoğraf yüklendi → **202** → işler `done` oldu → `GET /photos` analizli döndü
- [ ] `curl http://127.0.0.1:6333/collections` → `faces`, `identity_pool`, `photo_semantic`

---

## Yaygın Hatalar ve Çözümleri

### K1. `ERROR: No matching distribution found for torch==2.5.1+cu121`

**Neden:** `+cu121` yerel sürüm etiketi PyPI'da yok.

**Çözüm:** PyTorch'u önce kendi indeksinden kurun:
```powershell
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
GPU yoksa: [Adım 3.2](#32-cpu-only-kurulum-gpu-yoksa).

---

### K2. `pydantic_core.ValidationError: 4 validation errors for Settings` (başlangıçta)

**Örnek:**
```
DATABASE_URL  Field required
VLM_BASE_URL  Field required
VLM_MODEL     Field required
JWT_SECRET    Field required
```

**Neden:** `.env` yok, ya da **yanlış çalışma dizininden** başlattınız.
`SettingsConfigDict(env_file=".env")` `.env`'i **geçerli çalışma dizininde** arar.

**Çözüm:** `.env`'in var olduğundan ve komutları **repository kökünden**
çalıştırdığınızdan emin olun:
```powershell
cd C:\Users\<kullanici>\Desktop\photoai-backend
Get-Item .env
```

> `AI_PROVIDER=bedrock` kullanıyor olsanız bile `VLM_BASE_URL` ve `VLM_MODEL`
> **doldurulmalıdır** — bu alanların varsayılanı yoktur.

---

### K3. `FileNotFoundError: YuNet model dosyasi bulunamadi: models/face_detection_yunet_2023mar.onnx`

**Neden:** Model indirilmemiş veya yanlış klasörde; ya da worker'ı repository
kökü dışından çalıştırdınız (yol **görecelidir**).

**Çözüm:** [Adım 6.1](#61-yunet--yüz-tespiti-233-kb). Worker'ı her zaman
repository kökünden başlatın.

Dosya 1 KB civarındaysa Git LFS işaretçisi inmiştir → `curl -L` ile tekrar indirin.

---

### K4. `FileNotFoundError: AuraFace embedding modeli bulunamadi: models\auraface\glintr100.onnx`

**Çözüm:** [Adım 6.2](#62-auraface--yüz-embedding-261-mb).
```powershell
hf download fal/AuraFace-v1 glintr100.onnx --local-dir models\auraface
```

---

### K5. `/health` → `"qdrant": {"status": "error", ...}` veya bağlantı zaman aşımı

**Kontrol listesi:**
1. `qdrant.exe` çalışıyor mu? → `curl.exe http://127.0.0.1:6333/collections`
2. `.env`'de `QDRANT_URL=http://127.0.0.1:6333` mi? **`localhost` yazmayın** —
   Windows'ta IPv6 çözümlemesi her çağrıya ~2 sn ekler.
3. Qdrant **kendi klasöründen** mi başlatıldı? Başka bir dizinden başlatılırsa
   boş bir `.\storage` deposu açar ve **tüm yüz vektörleri kaybolmuş görünür**.

---

### K6. `RuntimeError: Qdrant 'photo_semantic' koleksiyonunun vektor boyutu 768, ayar (EMBEDDING_DIM) 1024`

**Neden:** `EMBEDDING_PROVIDER` / `EMBEDDING_DIM` değişmiş ama koleksiyon eski
boyutta. `ensure_collections()` bunu **uygulama başlangıcında** reddeder (yanlış
boyutlu vektör upsert'i Qdrant tarafında sessizce bozulur).

**Çözüm:**
```powershell
curl.exe -X DELETE http://127.0.0.1:6333/collections/photo_semantic
# backend'i yeniden başlatın (koleksiyon yeni boyutta oluşur), sonra:
python scripts\backfill_photo_semantic.py --reindex-all
```

---

### K7. `RuntimeError: Embedding provider boyutu (768) ile ayar EMBEDDING_DIM (1024) uyusmuyor`

**Neden:** `EMBEDDING_PROVIDER=local` (e5-base, sabit 768) iken `EMBEDDING_DIM`
başka bir değere ayarlanmış.

**Çözüm:** `local` için `EMBEDDING_DIM=768`. Titan (`bedrock`) için
**256 / 512 / 1024**'ten biri.

---

### K8. `RuntimeError: e5 model boyutu ..., beklenen 768 - yanlis model dizini olabilir`

**Neden:** `EMBEDDING_MODEL_DIR` yanlış modele işaret ediyor veya klasör eksik
indirilmiş (`modules.json` / `1_Pooling/` yoksa sentence-transformers modeli
düz transformer gibi yükler).

**Çözüm:** Klasörü silip tam olarak yeniden indirin:
```powershell
Remove-Item -Recurse -Force models\multilingual-e5-base
hf download intfloat/multilingual-e5-base --local-dir models\multilingual-e5-base
```

---

### K9. Worker hemen kapanıyor: `HATA: JOB_TYPES tanimsiz ya da bos`

**Neden:** Bu **kasıtlı bir fail-fast'tir** — varsayılan "tümünü tüket" davranışı yoktur.

**Çözüm:** PowerShell'de `set` **çalışmaz**, `$env:` kullanın:
```powershell
$env:JOB_TYPES="vlm_analysis"; python -m app.worker.main
```
cmd.exe'de:
```bat
set JOB_TYPES=vlm_analysis && python -m app.worker.main
```

---

### K10. Yüklenen fotoğraflar sonsuza kadar `queued` durumunda

**Neden ve sırayla kontrol:**
1. **Worker'lar çalışıyor mu?** → `/health` → `workers.*.status`
   `down` ise o tipte hiç worker başlamamıştır.
2. **Ingestion çalışıyor mu?** → `/health` → `ingestion.inbox.ready`
   Sürekli büyüyorsa döngü takılmıştır; backend loglarına bakın.
3. `INGESTION_ENABLED=false` mı? O zaman `python -m app.ingestion.main`
   ayrıca çalıştırılmalıdır.

---

### K11. `/health` → `"status": "degraded"`, `workers.semantic_index: {"status": "down"}`

**Neden:** Üçüncü worker (`semantic_index`) başlatılmamış. Bu worker olmadan
yeni yüklenen fotoğraflar **semantik aramada çıkmaz**.

**Çözüm:**
```powershell
$env:JOB_TYPES="semantic_index"; python -m app.worker.main
```
`SEMANTIC_SEARCH_ENABLED=false` ile özelliği tamamen kapatabilirsiniz (o zaman
`down` durumu yine görünür ama işlevsel etkisi olmaz).

---

### K12. Tarayıcıda CORS hatası: *"has been blocked by CORS policy"*

**Neden:** `CORS_ORIGINS` frontend'in **tam adresini** içermiyor.
`allow_credentials=True` olduğu için `*` **kullanılamaz**.

**Çözüm:** Şema + host + port birebir yazılmalı:
```ini
CORS_ORIGINS=http://192.168.1.50:5173,http://localhost:5173
```
Backend'i yeniden başlatın.

---

### K13. Frontend backend'e ulaşamıyor / 404

**Neden:** Backend **8001** portunda, frontend 8000'e bakıyor.

**Çözüm:** Frontend'in `.env` dosyasında `VITE_API_URL=http://localhost:8001`.

---

### K14. `/health` → `"vlm": {"status": "error", ...}`

**`AI_PROVIDER=lm_studio` ise:**
- LM Studio açık ve **Local Server başlatılmış** mı?
- Model yüklü mü? → `curl.exe http://localhost:1234/v1/models`
- `VLM_BASE_URL` `/v1` ile mi bitiyor?

**`AI_PROVIDER=bedrock` ise:**
- `aws sts get-caller-identity` çalışıyor mu?
- `AWS_REGION` doğru mu?
- Bedrock konsolunda ilgili modele **model access** verilmiş mi?

> Bedrock modunda `/health` modeli **çağırmaz**, yalnızca STS ile kimliği
> doğrular — yani `ok` görmek "InvokeModel yapabiliyoruz" garantisi değildir.

---

### K15. `QueuePool limit of size 10 overflow 15 reached` / rastgele 500 hataları

**Neden:** Eş zamanlı yükleme penceresi + durum yoklaması havuz tavanını zorluyor.
Havuz `app/db/session.py`'de **bilinçli olarak** 10 + 15 = 25 ayarlıdır.

**Çözüm:** Frontend'in eş zamanlı yükleme sayısını düşürün, ya da
`session.py`'deki `pool_size`/`max_overflow` değerlerini artırın —
PostgreSQL'in `max_connections` (varsayılan 100) sınırını unutmayın; worker
süreçleri de **kendi havuzlarını** açar.

---

### K16. `ValueError: IDENTITY_SEARCH_BACKEND=pg_brute_force PR-D bitene kadar ACILAMAZ`

**Neden:** Bu bayrak **kasıtlı olarak kilitlidir**. `merge_identities` /
`label_cluster` / `reassign_face` / `delete_photo` halen yalnızca Qdrant'a
yazıyor; erken açılırsa birleştirilmiş kimlikler PG brute-force aramasında
**yeniden ayrışır**.

**Çözüm:** `IDENTITY_SEARCH_BACKEND=qdrant` bırakın (varsayılan).

---

### K17. Testler üretim verisini kirletiyor / `test_worker_heartbeat.py` sürekli kalıyor

**Neden:** `PHOTOAI_TEST_USE_MAIN_DB=1` ayarlanmış olabilir. Bu bayrak testleri
paylaşımlı üretim DB'sine yönlendirir; canlı worker'lar global canlılık
assert'lerini bozar.

**Çözüm:** Bayrağı kaldırın — varsayılan davranış `photoai_db_test`
veritabanını otomatik oluşturur.

---

### K18. `Set-ExecutionPolicy` hatası: `Activate.ps1 cannot be loaded`

**Çözüm:**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
veya `cmd.exe` içinde `venv\Scripts\activate.bat` kullanın.

---

### K19. Yüz tanıma çok yavaş (fotoğraf başına 2 sn+)

**Neden:** `--reload` proje kökünü izliyor; `venv\` içindeki ~44.000 dosya
sürekli taranıyor.

**Çözüm:** `--reload-dir app` ekleyin (ölçüm: 1,97 sn → 0,29 sn, ~6 kat) veya
üretimde `--reload` kullanmayın.

---

### K20. Worker loglarında `KILIT CAKISMASI ESIGI ASILDI ... SIZMIS ADVISORY LOCK SUPHESI`

**Neden:** Bir iş `JOB_LOCK_CONFLICT_ESCALATION_ATTEMPTS` (10) kez lock-conflict
requeue yaşadı. Bu **otomatik fail etmez** — sızmış bir advisory lock şüphesi
sinyalidir (worker crash sonrası havuza kilit tutan bir bağlantı dönmüş olabilir).

**Önerilen müdahale (log mesajının kendisinden):**
1. İlgili worker süreçlerini yeniden başlatın (havuzdaki sızmış bağlantıyı düşürür).
2. Gerekirse inceleyin:
   ```powershell
   psql -U postgres -d photoai_db -c "SELECT * FROM pg_locks WHERE locktype='advisory';"
   ```

---

### K21. `uploads/inbox` içinde biriken `.part` dosyaları

**Neden:** İstek ortasında backend kapandı (yarım yükleme).

**Davranış:** `INGESTION_STALE_PART_SECONDS` (1800 sn = 30 dk) boyunca
dokunulmamış `.part` dosyaları "ölü yükleme" sayılır ve `uploads/failed/`
altına süpürülür. Yavaş ağ üzerinden büyük dosya yüklüyorsanız bu değeri
artırın.

**İzleme:** `/health` → `ingestion.inbox.in_progress` ve `.failed`.

---

### K22. Port 8000 "hayalet dinleyici" olarak takılı kalıyor

**Neden:** Windows'ta bilinen bir durum — `taskkill` / `Get-Process` ile
temizlenemiyor, yalnızca yeniden başlatma çözüyor.

**Çözüm:** Backend zaten **kalıcı olarak 8001**'e taşınmıştır. 8000 kullanmayın.

---

## Ek: Bilinen Sınırlamalar (README'den)

- `WORKER_FACE_PROCESSES` şu an **1** ile sınırlı: birden fazla yüz-hattı
  worker'ı aynı kimliğin paylaşılan centroid durumunu eşzamanlı güncelleyebilir.
  Artırmadan önce `test_identity_locks.py`, `test_face_pipeline_concurrency.py`,
  `test_worker_lock_escalation.py` ve `jobs_repository.py` başındaki not
  gözden geçirilmelidir.
- `sequence_in_user` hiç sıfırlanmaz; `priority` alanında yaşlanma/starvation
  önleme mekanizması yoktur — kabul edilmiş, bilinçli sınırlamalar.

---

## Ek: Dış Kaynak Bağlantıları

| Kaynak | Bağlantı |
|---|---|
| Python 3.11.9 | https://www.python.org/downloads/release/python-3119/ |
| PostgreSQL (Windows) | https://www.postgresql.org/download/windows/ |
| PostgreSQL installer (EDB) | https://www.enterprisedb.com/downloads/postgres-postgresql-downloads |
| Qdrant kurulum dokümantasyonu | https://qdrant.tech/documentation/guides/installation/ |
| Qdrant sürümleri (Windows binary) | https://github.com/qdrant/qdrant/releases |
| PyTorch önceki sürümler (cu121) | https://pytorch.org/get-started/previous-versions/ |
| PyTorch cu121 tekerlek indeksi | https://download.pytorch.org/whl/cu121 |
| YuNet modeli (OpenCV Zoo, MIT) | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet |
| AuraFace-v1 (Apache-2.0) | https://huggingface.co/fal/AuraFace-v1 |
| multilingual-e5-base (MIT) | https://huggingface.co/intfloat/multilingual-e5-base |
| Hugging Face CLI (`hf download`) | https://huggingface.co/docs/huggingface_hub/guides/cli |
| LM Studio | https://lmstudio.ai/ |
| LM Studio OpenAI-uyumlu API | https://lmstudio.ai/docs/app/api/endpoints/openai |
| AWS Bedrock kullanıcı kılavuzu | https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html |
| AWS Bedrock model erişimi | https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html |
| AWS kimlik bilgisi yapılandırması | https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html |
| FastAPI | https://fastapi.tiangolo.com/ |
| Alembic | https://alembic.sqlalchemy.org/en/latest/ |
| SQLAlchemy 2.0 | https://docs.sqlalchemy.org/en/20/ |
