import uuid
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.db.models import Photo, PhotoAnalysis
from app.ai.dispatcher import analyze_photo
from app.config import settings

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def save_upload(db: Session, file: UploadFile) -> Photo:
    """Dosyayı diske yazar, photos kaydını oluşturur."""
    ext = Path(file.filename).suffix.lower()
    photo_id = uuid.uuid4()
    stored_name = f"{photo_id}{ext}"
    storage_path = UPLOAD_DIR / stored_name

    with open(storage_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    photo = Photo(
        id=photo_id,
        filename=file.filename,
        storage_path=str(storage_path),
        status="processing",
    )
    db.add(photo)
    db.commit()
    db.refresh(photo)
    return photo


async def run_analysis(db: Session, photo: Photo) -> Photo:
    """VLM'i senkron çağırır, sonucu photo_analysis'e yazar."""
    try:
        result = await analyze_photo(photo.storage_path)

        analysis = PhotoAnalysis(
            photo_id=photo.id,
            caption=result.caption,
            environment=result.environment,
            activity=result.activity,
            people_count=result.people_count,
            possible_event=result.possible_event,
            summary=result.summary,
            model_name=settings.VLM_MODEL,
            analyzed_at=datetime.utcnow(),
        )
        db.add(analysis)
        photo.status = "analyzed"
    except Exception as e:
        print(f"[ANALIZ HATASI] {photo.filename}: {type(e).__name__}: {e}")
        photo.status = "failed"

    db.commit()
    db.refresh(photo)
    return photo


def get_photo_with_analysis(db: Session, photo_id: uuid.UUID):
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        return None, None
    analysis = db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == photo_id).first()
    return photo, analysis


def list_photos(db: Session):
    photos = db.query(Photo).order_by(Photo.created_at.desc()).all()
    out = []
    for p in photos:
        a = db.query(PhotoAnalysis).filter(PhotoAnalysis.photo_id == p.id).first()
        out.append((p, a))
    return out