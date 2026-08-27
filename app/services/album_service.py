"""Albüm CRUD + fotoğraf ekleme/çıkarma - BACKEND_IHTIYACLARI.md #1.

Basit bir join-tablosu üzerinden çalışır (`AlbumPhoto`) - kimlik
(person/cluster) tarafındaki kilit/Qdrant/centroid karmaşıklığının
HİÇBİRİ burada yok, bu yüzden bu servis kasıtlı olarak çok daha sade.
"""

import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.time import to_iso_utc
from app.db.models import Album, AlbumPhoto, Photo
from app.services import activity_log_service


def album_to_dict(db: Session, album: Album) -> dict:
    photo_count = (
        db.query(func.count(AlbumPhoto.photo_id)).filter(AlbumPhoto.album_id == album.id).scalar() or 0
    )
    return {
        "album_id": str(album.id),
        "name": album.name,
        "description": album.description,
        "cover_photo_id": str(album.cover_photo_id) if album.cover_photo_id else None,
        "photo_count": photo_count,
        "created_by_user_id": str(album.created_by_user_id) if album.created_by_user_id else None,
        "created_at": to_iso_utc(album.created_at),
        "updated_at": to_iso_utc(album.updated_at),
    }


def list_albums(db: Session) -> list[dict]:
    albums = db.query(Album).order_by(Album.created_at.desc()).all()
    return [album_to_dict(db, a) for a in albums]


def get_album(db: Session, album_id: uuid.UUID) -> Album:
    album = db.get(Album, album_id)
    if album is None:
        raise ValueError("Albüm bulunamadı")
    return album


def create_album(db: Session, name: str, description: str | None, created_by_user_id: uuid.UUID | None) -> Album:
    album = Album(
        id=uuid.uuid4(),
        name=name,
        description=description,
        created_by_user_id=created_by_user_id,
        created_at=datetime.utcnow(),
    )
    db.add(album)
    db.flush()
    activity_log_service.log(db, created_by_user_id, "album_create", "album", album.id, name)
    db.commit()
    db.refresh(album)
    return album


def update_album(
    db: Session,
    album_id: uuid.UUID,
    name: str | None,
    description: str | None,
    cover_photo_id: uuid.UUID | None,
    actor_user_id: uuid.UUID | None = None,
) -> Album:
    album = get_album(db, album_id)

    if name is not None:
        album.name = name
    if description is not None:
        album.description = description
    if cover_photo_id is not None:
        # Kapak, albümde OLMASA bile herhangi bir gerçek fotoğraf olabilir -
        # kısıtlama sahte bir kural olurdu (mockup'ta böyle bir kısıt yok).
        # Yalnızca fotoğrafın GERÇEKTEN var olduğu doğrulanır.
        if db.get(Photo, cover_photo_id) is None:
            raise ValueError("Kapak fotoğrafı bulunamadı")
        album.cover_photo_id = cover_photo_id

    album.updated_at = datetime.utcnow()
    db.add(album)
    activity_log_service.log(db, actor_user_id, "album_update", "album", album.id, album.name)
    db.commit()
    db.refresh(album)
    return album


def delete_album(db: Session, album_id: uuid.UUID, actor_user_id: uuid.UUID | None = None) -> dict:
    """Albümü siler - FOTOĞRAFLARA DOKUNMAZ (yalnızca `album_photos`
    kayıtları DB-level CASCADE ile gider), kişi/küme silmedeki ilkeyle
    AYNI (bkz. person_service.delete_identity)."""
    album = get_album(db, album_id)
    deleted_name = album.name
    db.delete(album)
    activity_log_service.log(db, actor_user_id, "album_delete", "album", album_id, deleted_name)
    db.commit()
    return {"deleted_album_id": str(album_id)}


def add_photos(db: Session, album_id: uuid.UUID, photo_ids: list[uuid.UUID], added_by_user_id: uuid.UUID | None) -> dict:
    """POST /albums/{id}/photos - Fotoğraflar sayfasındaki çoklu seçimden
    TOPLU ekleme. Var olmayan fotoğraf id'leri ve ZATEN albümde olanlar
    sessizce atlanır (409/500 ile tüm isteği PATLATMAK yerine - kısmi
    başarı ilkesiyle AYNI, bkz. GEREKSINIMLER.md)."""
    album = get_album(db, album_id)

    valid_photo_ids = {row[0] for row in db.query(Photo.id).filter(Photo.id.in_(photo_ids)).all()}
    already_in_album = {
        row[0]
        for row in db.query(AlbumPhoto.photo_id)
        .filter(AlbumPhoto.album_id == album_id, AlbumPhoto.photo_id.in_(photo_ids))
        .all()
    }

    added = 0
    first_added_photo_id: uuid.UUID | None = None
    for photo_id in photo_ids:
        if photo_id not in valid_photo_ids or photo_id in already_in_album:
            continue
        db.add(AlbumPhoto(album_id=album_id, photo_id=photo_id, added_by_user_id=added_by_user_id, added_at=datetime.utcnow()))
        added += 1
        if first_added_photo_id is None:
            first_added_photo_id = photo_id

    # Albümün henüz kapağı yoksa, ilk gerçekten eklenen fotoğraf otomatik
    # kapak olur - mockup'taki "albüm kartı bir thumbnail gösterir"
    # beklentisini bir sonraki adıma (kullanıcının PATCH ile elle seçmesi)
    # kadar karşılar.
    if album.cover_photo_id is None and first_added_photo_id is not None:
        album.cover_photo_id = first_added_photo_id
        db.add(album)

    # TEK ozet satiri (fotograf basina AYRI DEGIL) - toplu ekleme 500
    # fotografa kadar cikabilir, feed'i bogmasin diye (bkz. yukaridaki
    # "kismi basari" ilkesi).
    if added > 0:
        activity_log_service.log(
            db, added_by_user_id, "album_photo_add", "album", album_id, album.name,
            extra={"added": added},
        )
    db.commit()
    return {
        "album_id": str(album_id),
        "requested": len(photo_ids),
        "added": added,
        "skipped": len(photo_ids) - added,
    }


def remove_photo(db: Session, album_id: uuid.UUID, photo_id: uuid.UUID, actor_user_id: uuid.UUID | None = None) -> dict:
    album_for_log = get_album(db, album_id)  # 404 tetikler, albüm yoksa

    row = (
        db.query(AlbumPhoto)
        .filter(AlbumPhoto.album_id == album_id, AlbumPhoto.photo_id == photo_id)
        .first()
    )
    if row is None:
        raise ValueError("Fotoğraf bu albümde değil")
    db.delete(row)

    album = db.get(Album, album_id)
    if album is not None and album.cover_photo_id == photo_id:
        # Kapak fotoğraf albümden çıkarıldı - kalan bir fotoğraf varsa onu
        # otomatik kapak yap, yoksa kapaksız bırak (albüm boş kaldı).
        remaining = (
            db.query(AlbumPhoto.photo_id)
            .filter(AlbumPhoto.album_id == album_id, AlbumPhoto.photo_id != photo_id)
            .first()
        )
        album.cover_photo_id = remaining[0] if remaining else None
        db.add(album)

    photo = db.get(Photo, photo_id)
    activity_log_service.log(
        db, actor_user_id, "album_photo_remove", "album", album_id, album_for_log.name,
        extra={"photo_filename": photo.filename if photo else None},
    )
    db.commit()
    return {"album_id": str(album_id), "removed_photo_id": str(photo_id)}


def get_album_photo_ids(db: Session, album_id: uuid.UUID) -> list[uuid.UUID]:
    get_album(db, album_id)  # 404 tetikler, albüm yoksa
    rows = (
        db.query(AlbumPhoto.photo_id)
        .filter(AlbumPhoto.album_id == album_id)
        .order_by(AlbumPhoto.added_at.desc())
        .all()
    )
    return [r[0] for r in rows]
