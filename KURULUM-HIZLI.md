# PhotoAI Backend — Hızlı Kurulum

Sadece komutlar. Açıklamalar ve hata çözümleri için [`KURULUM.md`](KURULUM.md).
Varsayılan yol: **GPU + LM Studio**. CPU/Bedrock alternatifleri `#` ile işaretli.

Ön koşul (elle, bir kere): Python 3.11.x, Git, PostgreSQL 13+, (GPU'lu ise)
güncel NVIDIA sürücüsü kurulu olmalı.

## Otomatik (önerilen)

Aşağıdaki adımların çoğunu (2–8) tek script yapar — [`setup.bat`](setup.bat)'a
çift tıklayın ya da:

```powershell
cd C:\Users\<kullanici>\Desktop\photoai-backend
.\setup.bat
```

İdempotenttir: zaten yapılmış adımları atlar, mevcut `.env`'in üzerine
yazmaz. `.env` içindeki `DATABASE_URL` ve `JWT_SECRET` otomatik doldurulur;
`AI_PROVIDER`/`VLM_BASE_URL`/`VLM_MODEL` LM Studio varsayılanlarıyla gelir —
Bedrock kullanacaksanız `.env`'i script sonrasında elle düzenleyin (AWS
kimlik bilgileri script tarafından hiç okunmaz/dokunulmaz).

Parametreler: `setup.bat -Cpu` (CUDA yerine CPU-only), `-SkipQdrant`,
`-SkipDb`, `-SkipModels` (zaten kuruluysa), `-Lan` (LAN erişimi için
`CORS_ORIGINS`'i de sorar). Detay: `setup.ps1` başındaki yorum bloğu.

Script'in **yapmadıkları** (elle gerekir): PostgreSQL/Git/Python kurulumunun
kendisi, LM Studio'da model indirme, `aws configure`, servisleri başlatma
(bkz. [Adım 10](#10-başlatma--4-ayrı-pencere)).

## Elle, adım adım

---

## 1) Repo + sanal ortam

```powershell
cd C:\Users\<kullanici>\Desktop
git clone <repository-url> photoai-backend
cd photoai-backend

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Bağımlılıklar

```powershell
# GPU (CUDA 12.1):
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# --- CPU-only ise yukarıdaki 2 satır yerine ---
# (Get-Content requirements.txt) -replace 'torch==2\.5\.1\+cu121', 'torch==2.5.1' | Set-Content requirements-cpu.txt -Encoding utf8
# pip install -r requirements-cpu.txt

# doğrulama:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3) PostgreSQL

```powershell
psql -U postgres -c "CREATE DATABASE photoai_db;"
```

## 4) Qdrant

```powershell
New-Item -ItemType Directory -Force C:\qdrant
# zip'i C:\qdrant altına açın: https://github.com/qdrant/qdrant/releases (qdrant-x86_64-pc-windows-msvc.zip)
cd C:\qdrant
.\qdrant.exe
# --- yeni pencerede doğrulama ---
# curl.exe http://127.0.0.1:6333/collections
```

## 5) Model dosyaları (~1,4 GB)

```powershell
cd C:\Users\<kullanici>\Desktop\photoai-backend
New-Item -ItemType Directory -Force models\auraface

curl.exe -L -o models\face_detection_yunet_2023mar.onnx `
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

hf download fal/AuraFace-v1 glintr100.onnx --local-dir models\auraface

hf download intfloat/multilingual-e5-base --local-dir models\multilingual-e5-base
```

## 6) `.env`

```powershell
copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # JWT_SECRET icin
notepad .env
```

`.env` içinde minimum doldurulacaklar (LM Studio + GPU varsayılanı):

```ini
DATABASE_URL=postgresql://postgres:<PAROLA>@localhost:5432/photoai_db
AI_PROVIDER=lm_studio
VLM_BASE_URL=http://localhost:1234/v1
VLM_MODEL=qwen/qwen2.5-vl-7b
JWT_SECRET=<uretilen-deger>
CORS_ORIGINS=http://localhost:5173
```

```ini
# --- Bedrock kullanılacaksa AI_PROVIDER, AWS_REGION, AWS_BEDROCK_MODEL_ID ---
# AI_PROVIDER=bedrock
# AWS_REGION=us-east-1
# AWS_BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
# aws configure    # AWS kimlik bilgileri .env'e YAZILMAZ
```

## 7) VLM sağlayıcısı

```powershell
# LM Studio: uygulamayı açıp Qwen2.5-VL-7B'yi indirip Local Server'ı başlatın (port 1234)
curl.exe http://localhost:1234/v1/models

# --- Bedrock ise ---
# aws sts get-caller-identity
```

## 8) Migration'lar

```powershell
alembic upgrade head
alembic current    # b4d7e2a9c153 (head) beklenir
```

## 9) İlk admin

```powershell
python scripts\create_admin.py --email admin@ornek.com --password <GUCLU_PAROLA> --display-name "Ad Soyad"
```

## 10) Başlatma — 4 ayrı pencere

```powershell
# Pencere 1 — Backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --reload-dir app --port 8001
```
```powershell
# Pencere 2 — worker-vlm
.\venv\Scripts\Activate.ps1
$env:JOB_TYPES="vlm_analysis"; python -m app.worker.main
```
```powershell
# Pencere 3 — worker-face
.\venv\Scripts\Activate.ps1
$env:JOB_TYPES="face_pipeline"; python -m app.worker.main
```
```powershell
# Pencere 4 — worker-semantic
.\venv\Scripts\Activate.ps1
$env:JOB_TYPES="semantic_index"; python -m app.worker.main
```

> Tek komutla hepsi: bir üst klasördeki `..\start-photoai.bat`.

## 11) Doğrulama

```powershell
curl.exe http://localhost:8001/health
# "status": "ok" — checks VE workers hepsi "ok" olmalı

curl.exe http://127.0.0.1:6333/collections
# faces, identity_pool, photo_semantic

# tarayıcı: http://localhost:8001/docs
```

---

Sorun çıkarsa → [`KURULUM.md`](KURULUM.md) "Yaygın Hatalar ve Çözümleri" (K1–K22).
