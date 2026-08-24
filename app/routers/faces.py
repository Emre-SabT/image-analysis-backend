import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_role
from app.db.models import User
from app.db.session import get_db
from app.routers.photos import _to_dict as _photo_to_dict
from app.schemas import (
    ClusterLabelRequest,
    FaceReassignRequest,
    IdentityMergeRequest,
    RejectMergeRequest,
)
from app.services import clustering_service, face_service, person_service, photo_service

# Okuma uclari icin taban gereksinim: gecerli, oturum acmis herhangi bir
# kullanici (viewer dahil). Mutasyon uclari asagida ayrica
# require_role("admin", "editor") ile korunur.
router = APIRouter(dependencies=[Depends(get_current_user)])


def _person_to_dict(person) -> dict:
    return {
        "person_id": str(person.id),
        "display_name": person.display_name,
        "cluster_id": str(person.cluster_id) if person.cluster_id else None,
        "face_count": person.face_count,
        "created_by_user_id": str(person.created_by_user_id) if person.created_by_user_id else None,
        "created_at": person.created_at.isoformat() if person.created_at else None,
    }


def _face_to_dict(face) -> dict:
    return {
        "face_id": str(face.id),
        "photo_id": str(face.photo_id),
        "cluster_id": str(face.cluster_id) if face.cluster_id else None,
        "person_id": str(face.person_id) if face.person_id else None,
        "assigned_by": face.assigned_by,
        "crop_path": face.crop_path,
    }


@router.get("/clusters")
def get_clusters(status: str = Query("unlabeled"), db: Session = Depends(get_db)):
    """GET /clusters?status=unlabeled - isimlendirme bekleyen kumeler (ornek yuzlerle)."""
    return person_service.list_clusters(db, status=status)


@router.get("/clusters/merge-suggestions")
def get_merge_suggestions(db: Session = Depends(get_db)):
    """HDBSCAN katmani: ayni kisiye ait olabilecek klasorleri bulup birlestirme
    ONERISI dondurur. Hicbir sey degistirmez - uygulama karari kullanicinin."""
    return clustering_service.suggest_merges(db)


@router.post("/clusters/{cluster_id}/label", dependencies=[Depends(require_role("admin", "editor"))])
def label_cluster(
    cluster_id: uuid.UUID,
    body: ClusterLabelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """POST /clusters/{id}/label - kumeye kisi adi atar (B2)."""
    try:
        person = person_service.label_cluster(db, cluster_id, body.display_name, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _person_to_dict(person)


@router.get("/persons")
def get_persons(db: Session = Depends(get_db)):
    """GET /persons - isimlendirilmis tum kisiler ("klasorler"), birer ornek yuzle."""
    return person_service.list_persons(db)


@router.get("/persons/{person_id}/photos")
def get_person_photos(person_id: uuid.UUID, db: Session = Depends(get_db)):
    """GET /persons/{id}/photos - bir kisinin yer aldigi tum fotograflar."""
    photos = person_service.get_photos_for_person(db, person_id)
    result = []
    for photo in photos:
        _, analysis = photo_service.get_photo_with_analysis(db, photo.id)
        faces = face_service.get_faces_for_photo(db, photo.id)
        result.append(_photo_to_dict(photo, analysis, faces))
    return result


@router.post("/identities/merge", dependencies=[Depends(require_role("admin", "editor"))])
def merge_identities(
    body: IdentityMergeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """POST /identities/merge - iki klasoru (kisi ya da isimsiz kume, herhangi
    bir kombinasyon) birlestirir (B4)."""
    try:
        result = person_service.merge_identities(
            db, body.target_kind, body.target_id, body.source_kind, body.source_id, current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/identities/reject-merge", dependencies=[Depends(require_role("admin", "editor"))])
def reject_merge(
    body: RejectMergeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """POST /identities/reject-merge - "bunlar ayni kisi degil" der; cannot_link
    kisiti yazilir ve ayni oneri bir daha uretilmez."""
    try:
        return person_service.reject_merge(
            db, [{"kind": i.kind, "id": i.id} for i in body.identities], current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/identities/{kind}/{identity_id}", dependencies=[Depends(require_role("admin", "editor"))])
def delete_identity(kind: str, identity_id: uuid.UUID, db: Session = Depends(get_db)):
    """DELETE /identities/{kind}/{id} - bir klasoru ve icindeki tum yuz
    kayitlarini kalici olarak siler (yanlis tespit temizligi + FR-13/KVKK).
    Fotograflara dokunmaz."""
    try:
        return person_service.delete_identity(db, kind, identity_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/faces/{face_id}/reassign", dependencies=[Depends(require_role("admin", "editor"))])
def reassign_face(
    face_id: uuid.UUID,
    body: FaceReassignRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """POST /faces/{id}/reassign - yuzu kumeden cikarir / baska kisiye atar."""
    try:
        face = person_service.reassign_face(db, face_id, body.person_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _face_to_dict(face)
