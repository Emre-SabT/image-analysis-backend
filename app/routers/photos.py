import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_role
from app.core.settings import settings
from app.core.time import to_iso_utc
from app.db import jobs_repository
from app.db.models import (
    JOB_TYPE_FACE_PIPELINE,
    JOB_TYPE_VLM_ANALYSIS,
    User,
)
from app.db.session import get_db
from app.services import face_service, photo_service, semantic_service

logger = logging.getLogger("photoai.routers.photos")

router = APIRouter(prefix="/photos", tags=["photos"])


def _user_ref(user: User | None) -> dict | None:
    """{id, display_name} - `user` None ise (coklu kullanici gecisinden ONCE
    yuklenmis eski fotograflar, bkz. Photo.uploaded_by_user_id yorumu) None
    doner, sahte bir "Sistem" adi UYDURULMAZ."""
    if not user:
        return None
    return {"id": str(user.id), "display_name": user.display_name}


def _exif_to_dict(exif) -> dict | None:
    """BACKEND_IHTIYACLARI.md #6 - `exif` yoksa (henuz okunmadi/dosyada hic
    EXIF olmadigi icin tum alanlari None) `None` doner, bos bir obje DEGIL -
    frontend'in "Teknik" sekmesi bunu ayirt eder."""
    if not exif:
        return None
    return {
        "camera_make": exif.camera_make,
        "camera_model": exif.camera_model,
        "lens_model": exif.lens_model,
        "aperture": exif.aperture,
        "shutter_speed": exif.shutter_speed,
        "iso": exif.iso,
        "focal_length": exif.focal_length,
        # BILEREK to_iso_utc() KULLANILMIYOR: EXIF DateTimeOriginal, kameranin
        # KENDI yerel saatidir (zaman dilimi bilgisi EXIF'te YOK) - server
        # UTC'siyle AYNI konvansiyon degil. UTC etiketi eklemek, kamera
        # kullaniciyla FARKLI saat diliminde oldugunda YENI bir hataya yol
        # acardi (bkz. app/core/time.py, "captured_at" istisnasi).
        "captured_at": exif.captured_at.isoformat() if exif.captured_at else None,
        "gps_latitude": exif.gps_latitude,
        "gps_longitude": exif.gps_longitude,
        "copyright": exif.copyright,
        "width_px": exif.width_px,
        "height_px": exif.height_px,
        "file_size_bytes": exif.file_size_bytes,
    }


def _to_dict(photo, analysis, faces=None, uploaded_by: User | None = None, exif=None):
    data = {
        "photo_id": str(photo.id),
        "filename": photo.filename,
        "status": photo.status,
        "created_at": to_iso_utc(photo.created_at),
        "uploaded_by": _user_ref(uploaded_by),
        "exif": _exif_to_dict(exif),
    }
    if analysis:
        data.update({
            "description": analysis.description,
            "primary_object": analysis.primary_object,
            "secondary_objects": analysis.secondary_objects,
            "environment": analysis.environment,
            "attributes": analysis.attributes,
            "action": analysis.action,
            "mood": analysis.mood,
            "use_case": analysis.use_case,
            "context": analysis.context,
            "style": analysis.style,
            "audience": analysis.audience,
            "public_figures": analysis.public_figures,
            "all_tags": analysis.all_tags,
            "model_name": analysis.model_name,
            # BACKEND_IHTIYACLARI.md #5: tek GERCEK zaman damgali olay
            # (yukleme haric) - icerik analizinin NE ZAMAN bittigi. `None`
            # olabilir (analiz henuz bitmemis/basarisiz) - frontend'in
            # "Islem gecmisi" sekmesi bunu KOSULLU gosterir, uydurmaz.
            "analyzed_at": to_iso_utc(analysis.analyzed_at),
        })
    data["faces"] = [
        {
            "face_id": str(f.id),
            "quality_score": f.quality_score,
            "person_id": str(f.person_id) if f.person_id else None,
            "assigned_by": f.assigned_by,
            "is_background": f.is_background,
            # BACKEND_IHTIYACLARI.md #4 (BE-4): tespit kutusu koordinatlari
            # artik API'de doner - DB'de zaten vardi (Face.bbox), yalnizca
            # yaniti KISITLAYAN bu satir eksikti. {x, y, w, h}, piksel
            # cinsinden, orijinal foto boyutuna gore (bkz. face_service).
            "bbox": f.bbox,
        }
        for f in (faces or [])
    ]
    return data


# NOT: bilincli olarak `def` (async degil). Icerideki is bloke edici
# (dosya yazma, hash, DB). `def` olunca FastAPI endpoint'i threadpool'da
# calistirir.
#
# UC ASAMALI MIMARI - bu uc, YALNIZCA ILK asamadir:
#
#   1. UPLOAD (burasi)  dosyayi uploads/inbox'a akis halinde yazar, atomic
#                       rename ile tamamlar. DB'ye HICBIR SEY yazmaz.
#   2. INGESTION        ayri dongu (app/ingestion/main.py) inbox'u izler,
#                       Photo + EXIF + 2 is kaydini TEK commit'te olusturur.
#   3. WORKER           is kuyrugunu (FOR UPDATE SKIP LOCKED) tuketir.
#
# Neden boyle: "202 dondum" ile "fotograf gercekten kalici" arasindaki
# butun belirsizlik kalkti. Bu uc, dosya DISKTE TAMAMLANMIS halde
# bulunmadan basari donmez; o andan itibaren fotograf tarayicidan
# TAMAMEN bagimsizdir - sekme kapansa, ag kopsa, backend yeniden baslasa
# bile ingestion onu bulup isler.
#
# ISTEK ICINDE AI YOK: yuz hatti, VLM ve semantik indeksleme worker
# sureclerinde calisir. 500+ fotografli toplu yuklemede istek icinde
# isleme, zaman asimi ve is kaybi demekti.
@router.post("", status_code=202, dependencies=[Depends(require_role("admin", "editor"))])
def upload_photo(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in photo_service.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen format: {ext}. Izin verilen: JPEG, PNG, WEBP, HEIC",
        )

    received = photo_service.receive_upload(db, file, current_user.id)

    if received.duplicate:
        # Ayni icerik (SHA-256) daha once yuklenmis. Hash zaten akis
        # sirasinda hesaplandigi icin bu kontrol EK MALIYETSIZ ve
        # kullanicinin "Zaten arsivde" yanitini ANINDA almasini saglar.
        # (Asil tekillik garantisi DB'deki UNIQUE(content_hash); bu bir
        # on elemedir - bkz. photo_service.receive_upload.)
        photo = received.existing
        _, analysis = photo_service.get_photo_with_analysis(db, photo.id)
        faces = face_service.get_faces_for_photo(db, photo.id)
        # Mevcut (ilk) fotografin YUKLEYICISI - `current_user` DEGIL: ayni
        # icerigi baska biri daha once yuklemis olabilir, gercek sahiplik
        # UYDURULMAZ.
        original_uploader = (
            db.query(User).filter(User.id == photo.uploaded_by_user_id).first()
            if photo.uploaded_by_user_id
            else None
        )
        exif = photo_service.get_exif(db, photo.id)
        data = _to_dict(photo, analysis, faces, uploaded_by=original_uploader, exif=exif)
        data["duplicate"] = True
        return data

    # face_job_id / vlm_job_id ARTIK DONMUYOR (tip tanimlarinda zaten
    # optional'di): is kayitlari ingestion asamasinda olusuyor, bu uc
    # onlarin id'sini goremez. Istemci ilerlemeyi GET /photos/status ile
    # izler - toplu yuklemede zaten oyle yapiyordu.
    return {
        "photo_id": str(received.photo_id),
        "filename": received.filename,
        "status": "received",
        "duplicate": False,
        "uploaded_by": _user_ref(current_user),
    }


# --- Durum uclari (asenkron kuyruk akisi) ------------------------------
#
# ONEMLI - ROTA SIRASI: "/photos/status" bu dosyadaki "/photos/{photo_id}..."
# rotalarindan ONCE tanimlanmali; aksi halde FastAPI "status" kelimesini bir
# photo_id (UUID) sanip 422 dondurur.
#
# Neden bu uclara ihtiyac var: photo.status alani YALNIZCA VLM'i temsil
# ediyor (yuz hatti bu alana hic dokunmuyor). Yuz ve VLM artik BAGIMSIZ iki
# job oldugu icin ikisinin durumu ayri ayri raporlanmali.


@router.get("/status", dependencies=[Depends(get_current_user)])
def photos_status_batch(ids: str, db: Session = Depends(get_db)):
    """Toplu durum sorgusu: GET /photos/status?ids=uuid1,uuid2,...

    Toplu yuklemede (500+ fotograf) fotograf basina AYRI istek atmak
    tarayicinin origin basina ~6 eszamanli baglanti sinirina takilirdi;
    tek istekte coklu sorgu bu yuzden gerekli. Istemci URL uzunluk siniri
    nedeniyle id listesini parcalar (bkz. frontend api/photos.js).
    """
    photo_ids = [i.strip() for i in ids.split(",") if i.strip()]
    statuses = jobs_repository.photo_job_statuses(photo_ids, session=db)
    return [
        {"photo_id": pid, **statuses[pid]}
        for pid in photo_ids
        if pid in statuses
    ]


@router.get("/search", dependencies=[Depends(get_current_user)])
def search_photos(q: str, limit: int = 200, db: Session = Depends(get_db)):
    """Semantik arama - VLM JSON analizinden uretilmis metin embedding'leri
    uzerinde vektor benzerligiyle siralar (bkz. semantic_service.search).

    ROTA SIRASI: "/photos/{photo_id}/..." rotalarindan ONCE tanimli olmali
    (bkz. yukaridaki "/photos/status" notu - ayni gerekce).

    Donus: [{"photo_id", "score"}] - skora gore azalan. SADECE id + skor
    (tam Photo DEGIL): istemci zaten GET /photos ile tum listeyi tutuyor,
    bu id kumesini kendi facet filtreleriyle daraltip siralar.

    Hata sozlesmesi (frontend bunlarda istemci-tarafli substring aramasina
    DUSER):
      503 - ozellik kapali (SEMANTIC_SEARCH_ENABLED=False)
      502 - embedding modeli / Qdrant erisilemez
    """
    if not settings.SEMANTIC_SEARCH_ENABLED:
        raise HTTPException(status_code=503, detail="Semantik arama kapali")

    limit = max(1, min(limit, settings.SEMANTIC_SEARCH_TOP_K))
    try:
        # Alaka esigi: SEMANTIC_SEARCH_MIN_SCORE altindaki zayif komsular
        # hic donulmez (sonuc listesi bos gelebilir - frontend bunu
        # "eslesme yok" olarak gosterir, istemci substring aramasina
        # DUSMEZ; o dusus yalnizca 502/503'te olur).
        hits = semantic_service.search(q, limit, settings.SEMANTIC_SEARCH_MIN_SCORE)
    except Exception as e:
        logger.warning("Semantik arama basarisiz (q=%r): %s: %s", q, type(e).__name__, e)
        raise HTTPException(
            status_code=502,
            detail=f"Semantik arama gecici olarak kullanilamiyor ({type(e).__name__})",
        )
    return [{"photo_id": pid, "score": score} for pid, score in hits]


@router.get("/{photo_id}/status", dependencies=[Depends(get_current_user)])
def photo_status(photo_id: uuid.UUID, db: Session = Depends(get_db)):
    """Tek fotografin yuz + VLM is durumu.

    Her iki alan da: queued | running | done | failed | absent
    Ikisi BAGIMSIZ - biri 'failed' olmasi digerinin degerini etkilemez.
    """
    statuses = jobs_repository.photo_job_statuses([str(photo_id)], session=db)
    return {"photo_id": str(photo_id), **statuses[str(photo_id)]}


@router.get("", dependencies=[Depends(get_current_user)])
def list_photos(db: Session = Depends(get_db)):
    pairs = photo_service.list_photos(db)
    # N+1 onlemek icin TUM yukleyicileri VE EXIF kayitlarini tek sorguda cek
    # (BACKEND_IHTIYACLARI.md - "kim yukledi" ve "#6 EXIF" artik GET /photos
    # yanitinda gerceklesiyor).
    user_ids = {p.uploaded_by_user_id for p, _ in pairs if p.uploaded_by_user_id}
    users_by_id = {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()} if user_ids else {}
    exif_by_photo_id = photo_service.get_exif_map(db, [p.id for p, _ in pairs])
    return [
        _to_dict(
            p, a, face_service.get_faces_for_photo(db, p.id),
            uploaded_by=users_by_id.get(p.uploaded_by_user_id),
            exif=exif_by_photo_id.get(p.id),
        )
        for p, a in pairs
    ]


@router.delete("/{photo_id}", dependencies=[Depends(require_role("admin", "editor"))])
def delete_photo(photo_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """DELETE /photos/{id} - fotografi ve ondan turemis tum veriyi siler.
    Fotograftaki yuzler de klasorlerinden cikarilir; etkilenen kisi/klasorlerin
    merkezleri kalan uyelerle yeniden hesaplanir."""
    try:
        return photo_service.delete_photo(db, photo_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{photo_id}/file", dependencies=[Depends(get_current_user)])
def get_photo_file(photo_id: uuid.UUID, db: Session = Depends(get_db)):
    photo, _ = photo_service.get_photo_with_analysis(db, photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Fotograf bulunamadi")
    path, media_type = photo_service.get_servable_file(photo)
    return FileResponse(path, media_type=media_type)


@router.get("/{photo_id}/faces/{face_id}/file", dependencies=[Depends(get_current_user)])
def get_face_crop_file(photo_id: uuid.UUID, face_id: uuid.UUID, db: Session = Depends(get_db)):
    face = face_service.get_face(db, face_id)
    if not face or face.photo_id != photo_id:
        raise HTTPException(status_code=404, detail="Yuz bulunamadi")
    if not Path(face.crop_path).exists():
        raise HTTPException(status_code=404, detail="Yuz kirpimi dosyasi bulunamadi")
    return FileResponse(face.crop_path, media_type="image/jpeg")