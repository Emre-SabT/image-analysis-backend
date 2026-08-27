"""Kurumsal ORTAK havuz etkinlik gunlugu (BACKEND_IHTIYACLARI.md #5 +
kullanici istegi - "bir albumde/kimlikte/fotografta yapilan TUM
islemlerden TUM kullanicilarin haberi olmali").

`log()` COMMIT YAPMAZ - cagiran servis fonksiyonunun KENDI transaction'ina
(zaten var olan `db.commit()`'ine) eklenir; boylece is mantigi basarisiz
olursa (ör. ValueError) gunluk satiri da geri alinir, YALNIZCA basarili
islemler gunluge girer.
"""

import uuid

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.time import to_iso_utc
from app.db.models import ActivityLog, User


def log(
    db: Session,
    actor_user_id: uuid.UUID | None,
    action: str,
    target_kind: str,
    target_id: uuid.UUID | None,
    target_label: str | None,
    extra: dict | None = None,
) -> None:
    db.add(
        ActivityLog(
            id=uuid.uuid4(),
            actor_user_id=actor_user_id,
            action=action,
            target_kind=target_kind,
            target_id=target_id,
            target_label=target_label,
            extra=extra,
        )
    )


def _user_ref(user: User | None) -> dict | None:
    if not user:
        return None
    return {"id": str(user.id), "display_name": user.display_name}


def list_activity(db: Session, limit: int = 50, photo_id: uuid.UUID | None = None) -> list[dict]:
    """GET /activity - en yeniden eskiye TUM kullanicilarin islemleri
    (kullanici bazli filtre YOK - kurumsal ortak havuz ilkesi, bkz. modul
    docstring'i).

    `photo_id` verilirse - Fotograf Detayi'nin "Islem Gecmisi" sekmesi icin
    (bkz. PhotoDetailHistoryTab.tsx) - yalnizca O FOTOGRAFI ilgilendiren
    olaylar donulur: dogrudan hedefi bu fotograf olanlar (`photo_upload`)
    VE `extra.photo_id`'si eslesenler (`face_reassign` - bkz. person_service.
    reassign_face). BILINCLI OLARAK KAPSAM DISI: `identity_merge`/
    `identity_reject_merge` bir KIMLIGI hedefler, o kimligin hangi
    fotograflardaki yuzleri kapsadigi bu satirda TUTULMUYOR - "bu
    fotograftaki bir yuz baska bir kimlikle birlestirildi" olayi burada
    GORUNMEYEBILIR (dogru olmayan bir sonuc UYDURMAKTANSA eksik birakildi).
    """
    query = db.query(ActivityLog)
    if photo_id is not None:
        query = query.filter(
            or_(
                (ActivityLog.target_kind == "photo") & (ActivityLog.target_id == photo_id),
                ActivityLog.extra["photo_id"].astext == str(photo_id),
            )
        )
    entries = query.order_by(ActivityLog.created_at.desc()).limit(limit).all()
    if not entries:
        return []

    user_ids = {e.actor_user_id for e in entries if e.actor_user_id}
    users_by_id = {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()} if user_ids else {}

    return [
        {
            "id": str(e.id),
            "actor": _user_ref(users_by_id.get(e.actor_user_id)),
            "action": e.action,
            "target_kind": e.target_kind,
            "target_id": str(e.target_id) if e.target_id else None,
            "target_label": e.target_label,
            "extra": e.extra,
            "created_at": to_iso_utc(e.created_at),
        }
        for e in entries
    ]
