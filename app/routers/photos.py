import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_role
from app.db import jobs_repository
from app.db.models import (
    JOB_TYPE_FACE_PIPELINE,
    JOB_TYPE_VLM_ANALYSIS,
    User,
)
from app.db.session import get_db
from app.services import face_service, photo_service

router = APIRouter(prefix="/photos", tags=["photos"])


def _to_dict(photo, analysis, faces=None):
    data = {
        "photo_id": str(photo.id),
        "filename": photo.filename,
        "status": photo.status,
        "created_at": photo.created_at.isoformat() if photo.created_at else None,
    }
    if analysis:
        data.update({
            "description": analysis.description,
            "environment_type": analysis.environment_type,
            "people_count": analysis.people_count,
            "possible_event": analysis.possible_event,
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
        })
    data["faces"] = [
        {
            "face_id": str(f.id),
            "quality_score": f.quality_score,
            "person_id": str(f.person_id) if f.person_id else None,
            "assigned_by": f.assigned_by,
            "is_background": f.is_background,
        }
        for f in (faces or [])
    ]
    return data


# NOT: bilincli olarak `def` (async degil). Icerideki is bloke edici
# (dosya yazma, hash, DB). `def` olunca FastAPI endpoint'i threadpool'da
# calistirir.
#
# ARTIK ISLEME YOK, YALNIZCA KUYRUGA ALMA: hem yuz hatti hem VLM analizi
# kalici kuyruga (jobs tablosu) yaziliyor ve ayri worker sureclerinde
# calisiyor. Onceki tasarimda yuz hatti istek ICINDE senkron calisiyordu;
# 500+ fotografli toplu yuklemede bu zaman asimi ve is kaybi demekti.
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

    photo, is_duplicate = photo_service.save_upload(db, file, current_user.id)

    if is_duplicate:
        # Ayni icerik (SHA-256) daha once yuklenmis: yeni is KUYRUGA ALINMAZ,
        # mevcut fotografin guncel hali donulur (ne yuz hatti ne VLM tekrar
        # calisir).
        db.commit()
        _, analysis = photo_service.get_photo_with_analysis(db, photo.id)
        faces = face_service.get_faces_for_photo(db, photo.id)
        data = _to_dict(photo, analysis, faces)
        data["duplicate"] = True
        return data

    # IKI BAGIMSIZ IS. Aralarinda sira/bagimlilik YOK, paralel calisabilirler;
    # biri fail olursa digeri etkilenmez ("yuz verisi VLM hatasindan
    # etkilenmez, kismi basari gecerlidir" ilkesi korunuyor).
    face_job_id = jobs_repository.enqueue(
        db, JOB_TYPE_FACE_PIPELINE, {"photo_id": str(photo.id)}, current_user.id
    )
    vlm_job_id = jobs_repository.enqueue(
        db, JOB_TYPE_VLM_ANALYSIS, {"photo_id": str(photo.id)}, current_user.id
    )

    # ATOMIK: foto satiri + 2 job satiri + 2 sayac artirimi TEK commit'te.
    # Job insert patlarsa foto insert de geri alinir (save_upload artik
    # commit degil flush yapiyor).
    db.commit()

    return {
        "photo_id": str(photo.id),
        "filename": photo.filename,
        "status": photo.status,
        "duplicate": False,
        "face_job_id": str(face_job_id),
        "vlm_job_id": str(vlm_job_id),
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
    return [
        _to_dict(p, a, face_service.get_faces_for_photo(db, p.id))
        for p, a in photo_service.list_photos(db)
    ]


@router.delete("/{photo_id}", dependencies=[Depends(require_role("admin", "editor"))])
def delete_photo(photo_id: uuid.UUID, db: Session = Depends(get_db)):
    """DELETE /photos/{id} - fotografi ve ondan turemis tum veriyi siler.
    Fotograftaki yuzler de klasorlerinden cikarilir; etkilenen kisi/klasorlerin
    merkezleri kalan uyelerle yeniden hesaplanir."""
    try:
        return photo_service.delete_photo(db, photo_id)
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