import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_role
from app.db.models import User
from app.db.session import get_db
from app.schemas import UserCreateRequest, UserResponse, UserUpdateRequest
from app.services import auth_service

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_role("admin"))])


@router.post("", response_model=UserResponse, status_code=201)
def create_user(payload: UserCreateRequest, db: Session = Depends(get_db)):
    return auth_service.create_user(
        db, payload.email, payload.password, payload.display_name, payload.role
    )


@router.get("", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    return auth_service.list_users(db)


@router.patch("/{user_id}", response_model=UserResponse)
def update_user(user_id: uuid.UUID, payload: UserUpdateRequest, db: Session = Depends(get_db)):
    return auth_service.update_user(
        db, user_id, payload.display_name, payload.role, payload.is_active
    )


@router.delete("/{user_id}", status_code=204)
def delete_user(
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """DELETE /users/{id} (BACKEND_IHTIYACLARI.md #8) - kalici silme.
    "Devre disi birak" (`PATCH .../{"is_active": false}`) hala VAR ve
    cogu senaryo icin TERCIH EDILEN yol (audit izi korunur) - bu, gercekten
    kalici silme istendiginde kullanilan, EK korumali (kendi hesabini
    silememe + is gecmisi varsa reddetme) ikinci bir uc."""
    auth_service.delete_user(db, user_id, current_user.id)
