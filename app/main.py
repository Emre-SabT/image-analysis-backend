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
from app.db import qdrant
from app.db.session import SessionLocal
from app.routers import auth, faces, jobs, photos, users

logger = logging.getLogger("photoai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collections()
    yield


app = FastAPI(title="PhotoAI", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(jobs.router)
app.include_router(photos.router)
app.include_router(faces.router)

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
    try:
        base = settings.VLM_BASE_URL.rstrip("/")
        url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        resp = httpx.get(url, timeout=3)
        resp.raise_for_status()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/health")
def health():
    checks = {
        "database": _check_database(),
        "qdrant": _check_qdrant(),
        "vlm": _check_vlm(),
    }
    overall = "ok" if all(c["status"] == "ok" for c in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
