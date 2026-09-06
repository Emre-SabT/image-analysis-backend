import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.exceptions import ServiceError, service_error_handler
from app.core.settings import settings
from app.db.qdrant import ensure_collections
from app.db import jobs_repository, qdrant
from app.db.session import SessionLocal
from app.ingestion.main import loop as ingestion_loop
from app.routers import albums, auth, faces, jobs, photos, system, users
from app.services import ingestion_service

logger = logging.getLogger("photoai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collections()
    # Inbox ingestion dongusu backend ile BIRLIKTE baslar (spec 15): ayri
    # bir servis/pencere yonetmeye gerek kalmadan, backend her yeniden
    # baslatildiginda inbox taranir, yarim kalmis islemler uzlastirilir ve
    # DB'ye alinmamis dosyalar islenmeye devam eder.
    if settings.INGESTION_ENABLED:
        ingestion_loop.start()
    else:
        logger.warning(
            "INGESTION_ENABLED=false - inbox dongusu BASLATILMADI. Yuklenen "
            "fotograflar uploads/inbox'ta birikir ve islenmez "
            "(ayri surec: python -m app.ingestion.main)."
        )
    try:
        yield
    finally:
        if settings.INGESTION_ENABLED:
            ingestion_loop.stop()


app = FastAPI(title="PhotoAI", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(jobs.router)
app.include_router(photos.router)
app.include_router(faces.router)
app.include_router(system.router)
app.include_router(albums.router)

# ServiceError (auth/kullanici hatalari - 401/403/404/409) -> tutarli JSON
# govdesi. Genel Exception handler'dan ONCE kaydedilir; Starlette daha
# spesifik olan bu handler'i tercih eder.
app.add_exception_handler(ServiceError, service_error_handler)


# --- Global hata yakalayici -------------------------------------------------
#
# Bundan once beklenmeyen bir hata FastAPI'nin varsayilan 500 yanitina
# duserdu - loglanmadan, tutarli bir govde formati olmadan. Artik her
# beklenmeyen hata loglanir (stack trace dahil) ve istemciye TEK TIP,
# ic detay sizdirmayan bir govde doner. Servis fonksiyonlarinin firlattigi
# ValueError'lar (router'larda zaten HTTPException'a cevriliyor) bu
# handler'a hic ugramiyor - sadece GERCEKTEN beklenmeyen (yakalanmamis)
# hatalar buraya duser.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Beklenmeyen hata: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Beklenmeyen bir sunucu hatasi olustu"},
    )


# --- /health - gercek bagimlilik kontrolleri --------------------------------
#
# Oncesinde her zaman {"status":"ok"} donuyordu - Postgres/Qdrant/VLM
# cokse bile. Artik ucu de AYRI AYRI kontrol edilip raporlaniyor; bir
# tanesinin cokmesi digerlerinin kontrolunu engellemez (her biri kendi
# try/except'inde).
def _check_database() -> dict:
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return {"status": "ok"}
        finally:
            db.close()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_qdrant() -> dict:
    try:
        qdrant.client.get_collections()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_vlm() -> dict:
    """AI_PROVIDER'a gore DOGRU yolu kontrol eder - dispatcher.py'nin
    IS_LM_STUDIO dallanmasiyla AYNI mantik. ONCEDEN AI_PROVIDER'dan BAGIMSIZ
    her zaman VLM_BASE_URL'e (LM Studio) bakiyordu - AI_PROVIDER=bedrock iken
    bu YANLIS sinyal veriyordu (LM Studio calisan biri varsa "ok" gorunurdu,
    asil kullanilan Bedrock hattinin durumuyla hicbir ilgisi olmadan)."""
    if settings.AI_PROVIDER == "lm_studio":
        try:
            base = settings.VLM_BASE_URL.rstrip("/")
            url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
            resp = httpx.get(url, timeout=3)
            resp.raise_for_status()
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # bedrock: gercek modeli CAGIRMAZ (ucretli + yavas, her /health isteginde
    # yapilamaz) - yalnizca AWS kimlik bilgilerinin GECERLI ve ERISILEBILIR
    # oldugunu dogrular (STS herhangi bir IAM principal icin izinli, ekstra
    # bedrock:* izni GEREKMEZ). Bu, "Bedrock'a InvokeModel yapabiliyor muyuz"
    # sorusunun TAM cevabi degil - "AWS'ye gecerli kimlikle baglanabiliyor
    # muyuz" sorusunun cevabidir; ama VLM_BASE_URL'e bakmaktan cok daha
    # dogru bir sinyal (o, bu saglayiciyla ALAKASIZ).
    try:
        import boto3
        from botocore.config import Config

        sts = boto3.client(
            "sts", region_name=settings.AWS_REGION, config=Config(connect_timeout=3, read_timeout=3)
        )
        sts.get_caller_identity()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _check_workers() -> dict:
    """Her BILINEN is tipi icin worker surecinin canliligi (worker_heartbeats
    tablosundan) - ok | stale | down. Bostaki worker'i da yakalar; kuyruk
    bos olsa bile "semantic_index worker calisiyor mu" sorusunu yanitlar.

    DB erisilemezse worker durumu da bilinemez - sessiz {} yerine acik
    hata dondurulur (cagiran degraded sayar)."""
    try:
        return jobs_repository.worker_liveness(settings.WORKER_HEARTBEAT_STALE_SECONDS)
    except Exception as e:
        return {"error": {"status": "error", "detail": str(e)}}


def _ingestion_report() -> dict:
    """Spec 14 - ingestion gozlemlenebilirligi. Mevcut /health zaten
    operatorun baktigi tek ekran (frontend HealthBanner + Sistem sayfasi);
    ayri bir metrik ucu ACMAK yerine oraya eklendi.

    `ready` surekli buyuyorsa dongu takilmis demektir; `orphan_meta` > 0
    ise commit/tasima arasinda bir crash yasanmis ve bir sonraki
    uzlastirmayi bekliyor demektir."""
    if not settings.INGESTION_ENABLED:
        return {"status": "disabled"}
    try:
        # IKI FARKLI KAVRAM, AYRI ISIMLER ALTINDA.
        #
        # Onceki hali ikisini duz birlestiriyordu ve `failed` anahtari
        # CAKISIYORDU: inbox_stats()["failed"] (failed/ klasorundeki DOSYA
        # sayisi) sessizce loop.stats["failed"] (surec basladigindan beri
        # KUMULATIF basarisiz ingest sayisi) tarafindan eziliyordu. Operator
        # "failed: 3" gorup klasore bakinca orayi BOS buluyordu.
        return {
            "status": "ok",
            # ANLIK durum - inbox/failed klasorlerinin o andaki icerigi
            "inbox": ingestion_service.inbox_stats(),
            # KUMULATIF sayaclar - bu surec basladigindan beri
            "totals": dict(ingestion_loop.stats),
            "last_reconcile_at": ingestion_loop.last_reconcile_at,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/health")
def health():
    checks = {
        "database": _check_database(),
        "qdrant": _check_qdrant(),
        "vlm": _check_vlm(),
    }
    # workers: {job_type: {status: ok|stale|down, workers, last_seen, seconds_since}}
    workers = _check_workers()
    deps_ok = all(c["status"] == "ok" for c in checks.values())
    workers_ok = all(w["status"] == "ok" for w in workers.values())
    overall = "ok" if deps_ok and workers_ok else "degraded"
    return {
        "status": overall,
        "checks": checks,
        "workers": workers,
        "ingestion": _ingestion_report(),
    }
