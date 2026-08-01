import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.config import settings
from app.services import photo_service

router = APIRouter(prefix="/photos", tags=["photos"])


def verify_service_key(x_service_key: str = Header(None)):
    if x_service_key != settings.SERVICE_KEY:
        raise HTTPException(status_code=401, detail="Gecersiz servis anahtari")


def _to_dict(photo, analysis):
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
    return data


@router.post("", dependencies=[Depends(verify_service_key)])
async def upload_photo(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename).suffix.lower()
    if ext not in photo_service.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen format: {ext}. Izin verilen: JPEG, PNG, WEBP, HEIC",
        )

    photo = photo_service.save_upload(db, file)
    photo = await photo_service.run_analysis(db, photo)
    _, analysis = photo_service.get_photo_with_analysis(db, photo.id)

    if photo.status == "failed":
        raise HTTPException(
            status_code=502,
            detail="AI analizi basarisiz oldu. Lutfen tekrar deneyin.",
        )

    return _to_dict(photo, analysis)


@router.get("", dependencies=[Depends(verify_service_key)])
def list_photos(db: Session = Depends(get_db)):
    return [_to_dict(p, a) for p, a in photo_service.list_photos(db)]


@router.get("/{photo_id}/file")
def get_photo_file(photo_id: uuid.UUID, db: Session = Depends(get_db)):
    photo, _ = photo_service.get_photo_with_analysis(db, photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Fotograf bulunamadi")
    path, media_type = photo_service.get_servable_file(photo)
    return FileResponse(path, media_type=media_type)