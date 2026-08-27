import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_role
from app.db.models import User
from app.db.session import get_db
from app.routers.photos import _to_dict as _photo_to_dict
from app.schemas import AlbumCreateRequest, AlbumPhotosAddRequest, AlbumUpdateRequest
from app.services import album_service, face_service, photo_service

# Okuma uclari icin taban gereksinim: gecerli, oturum acmis herhangi bir
# kullanici (viewer dahil) - faces.py'deki AYNI desen. Mutasyon uclari
# asagida ayrica require_role("admin", "editor") ile korunur.
router = APIRouter(prefix="/albums", tags=["albums"], dependencies=[Depends(get_current_user)])


@router.post("", status_code=201, dependencies=[Depends(require_role("admin", "editor"))])
def create_album(
    body: AlbumCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """POST /albums - `{name, description?}` → yeni, BOŞ albüm oluşturur."""
    album = album_service.create_album(db, body.name, body.description, current_user.id)
    return album_service.album_to_dict(db, album)


@router.get("")
def list_albums(db: Session = Depends(get_db)):
    """GET /albums - albüm listesi, her biri gerçek `photo_count` +
    `cover_photo_id` ile (kapak yoksa `null` - UYDURULMAZ)."""
    return album_service.list_albums(db)


@router.get("/{album_id}")
def get_album(album_id: uuid.UUID, db: Session = Depends(get_db)):
    try:
        album = album_service.get_album(db, album_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return album_service.album_to_dict(db, album)


@router.patch("/{album_id}", dependencies=[Depends(require_role("admin", "editor"))])
def update_album(
    album_id: uuid.UUID,
    body: AlbumUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """PATCH /albums/{id} - `{name?, description?, cover_photo_id?}`."""
    try:
        album = album_service.update_album(
            db, album_id, body.name, body.description, body.cover_photo_id, current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return album_service.album_to_dict(db, album)


@router.delete("/{album_id}", dependencies=[Depends(require_role("admin", "editor"))])
def delete_album(album_id: uuid.UUID, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """DELETE /albums/{id} - albümü siler, FOTOĞRAFLARA DOKUNMAZ (yalnızca
    `album_photos` kayıtları gider)."""
    try:
        return album_service.delete_album(db, album_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{album_id}/photos", dependencies=[Depends(require_role("admin", "editor"))])
def add_photos_to_album(
    album_id: uuid.UUID,
    body: AlbumPhotosAddRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """POST /albums/{id}/photos - `{photo_ids: UUID[]}` → TOPLU ekleme
    (Fotoğraflar sayfasının çoklu seçim akışı için tek istekte N
    fotoğraf). Geçersiz/zaten-albümde id'ler sessizce atlanır, sonuçta
    kaç tanesinin GERÇEKTEN eklendiği döner."""
    try:
        return album_service.add_photos(db, album_id, body.photo_ids, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{album_id}/photos/{photo_id}", dependencies=[Depends(require_role("admin", "editor"))])
def remove_photo_from_album(
    album_id: uuid.UUID,
    photo_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return album_service.remove_photo(db, album_id, photo_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{album_id}/photos")
def get_album_photos(album_id: uuid.UUID, db: Session = Depends(get_db)):
    """GET /albums/{id}/photos - bir albümdeki tüm fotoğraflar (tam
    fotoğraf gövdesiyle - `GET /persons/{id}/photos` ile AYNI desen,
    `_to_dict` paylaşılıyor)."""
    try:
        photo_ids = album_service.get_album_photo_ids(db, album_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = []
    for photo_id in photo_ids:
        photo, analysis = photo_service.get_photo_with_analysis(db, photo_id)
        if not photo:
            continue
        faces = face_service.get_faces_for_photo(db, photo.id)
        result.append(_photo_to_dict(photo, analysis, faces))
    return result
