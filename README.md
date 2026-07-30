# PhotoAI Backend

FastAPI tabanlı, yüklenen fotoğrafları bir VLM (vision-language model) ile analiz edip sonuçları PostgreSQL'de saklayan servis.

## Özellikler

- Fotoğraf yükleme (JPEG, PNG, WEBP, HEIC)
- Yüklenen fotoğrafın VLM ile analizi (caption, ortam, aktivite, kişi sayısı, olası etkinlik, özet)
- Analiz sonuçlarının veritabanında saklanması ve listelenmesi
- Servis anahtarı (`X-Service-Key` header) ile korunan uçlar

## Gereksinimler

- Python 3.11
- PostgreSQL

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Proje kökünde bir `.env` dosyası oluşturup aşağıdaki değişkenleri tanımlayın:

```
DATABASE_URL=postgresql://kullanici:sifre@localhost:5432/photoai
VLM_BASE_URL=http://localhost:1234/v1
VLM_MODEL=qwen/qwen2.5-vl-7b
SERVICE_KEY=degistir-bu-gizli-anahtari
AI_PROVIDER=lm_studio
```

> `SERVICE_KEY` burada tanımladığınız değer, API isteklerinde `X-Service-Key` header'ına aynen yazılmalıdır. Örn: `X-Service-Key: degistir-bu-gizli-anahtari`

## Veritabanı Migrasyonları

```bash
alembic upgrade head
```

## Çalıştırma

```bash
uvicorn app.main:app --reload
```

Servis varsayılan olarak `http://localhost:8000` adresinde çalışır.

Kurulumun doğru gittiğini kontrol etmek için:

```bash
curl http://localhost:8000/health
```

`{"status": "ok"}` benzeri bir cevap dönüyorsa servis ayaktadır.

## API Uçları

| Method | Endpoint            | Açıklama                                   | Auth              |
|--------|---------------------|---------------------------------------------|--------------------|
| GET    | `/health`            | Servis durumu kontrolü                     | -                  |
| POST   | `/photos`            | Fotoğraf yükler ve analiz eder             | `X-Service-Key`    |
| GET    | `/photos`            | Yüklenen fotoğrafları ve analizlerini listeler | `X-Service-Key` |
| GET    | `/photos/{photo_id}/file` | Fotoğraf dosyasını döner (dosya sunumu olduğu için genel erişime açık) | -    |

### Örnek: Fotoğraf Yükleme

```bash
curl -X POST http://localhost:8000/photos \
  -H "X-Service-Key: degistir-bu-gizli-anahtari" \
  -F "file=@ornek.jpg"
```

Örnek cevap:

```json
{
  "id": "3f2a1c90-1a2b-4c3d-9e8f-abc123456789",
  "filename": "ornek.jpg",
  "analysis": {
    "caption": "Bir grup insan sahilde gün batımını izliyor",
    "ortam": "dış mekan, sahil",
    "aktivite": "sosyalleşme",
    "kisi_sayisi": 4,
    "olasi_etkinlik": "gün batımı buluşması",
    "ozet": "Dört kişilik bir grup sahilde vakit geçiriyor"
  },
  "created_at": "2026-07-30T14:22:10Z"
}
```

### Örnek: Fotoğrafları Listeleme

```bash
curl http://localhost:8000/photos \
  -H "X-Service-Key: degistir-bu-gizli-anahtari"
```

Örnek cevap:

```json
[
  {
    "id": "3f2a1c90-1a2b-4c3d-9e8f-abc123456789",
    "filename": "ornek.jpg",
    "analysis": { "caption": "...", "..." : "..." },
    "created_at": "2026-07-30T14:22:10Z"
  }
]
```

## Hata Durumları

| Durum                                   | HTTP Kodu | Açıklama                                                        |
|------------------------------------------|-----------|-------------------------------------------------------------------|
| Geçersiz veya eksik `X-Service-Key`       | 401       | Header eksik ya da `.env`'deki `SERVICE_KEY` ile eşleşmiyor      |
| Desteklenmeyen dosya formatı              | 422       | `Body_upload_photo_photos_post` şemasına uymayan istek           |
| VLM servisi erişilemez (LM Studio kapalı) | 502 / 503 | Analiz adımı başarısız olur, fotoğraf yine de kaydedilebilir     |
| Var olmayan `photo_id`                    | 404       | `/photos/{photo_id}/file` uçları için                             |

## Proje Yapısı

```
app/
  ai/          # VLM dispatcher
  db/          # SQLAlchemy modelleri ve oturum yönetimi
  routers/     # API route tanımları
  services/    # İş mantığı (fotoğraf kaydetme, analiz)
  config.py    # Ortam değişkenleri
  main.py      # FastAPI uygulaması
alembic/       # Veritabanı migrasyonları
```