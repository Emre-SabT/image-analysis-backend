import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import require_role
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
