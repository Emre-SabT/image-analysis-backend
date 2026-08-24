from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas import LoginRequest, LogoutRequest, RefreshRequest, TokenResponse, UserResponse
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate(db, payload.email, payload.password)
    return auth_service.issue_tokens(db, user)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    return auth_service.rotate_refresh_token(db, payload.refresh_token)


@router.post("/logout")
def logout(
    payload: LogoutRequest,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    auth_service.revoke_refresh_token(db, payload.refresh_token)
    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    """Frontend'in giriş yapmış kullanıcının rolünü/adını öğrenmesi için
    (ör. Kullanıcılar sekmesini yalnızca admin'e göstermek)."""
    return current_user
