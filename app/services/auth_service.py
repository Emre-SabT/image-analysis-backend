# Kimlik dogrulama is mantigi: login, token yenileme/iptal, kullanici olusturma.
#
# Diger servislerle (person_service.py, photo_service.py) ayni desen: db bir
# Session olarak parametre gecirilir, fonksiyonlar duz senkron.

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationError, ConflictError, NotFoundError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.core.settings import settings
from app.db.models import RefreshToken, User
from app.schemas import TokenResponse


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def authenticate(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user or not verify_password(password, user.password_hash):
        raise AuthenticationError("E-posta ya da parola hatali")
    if not user.is_active:
        raise AuthenticationError("Kullanici hesabi devre disi")
    return user


def issue_tokens(db: Session, user: User) -> TokenResponse:
    access_token = create_access_token(user.id, user.role)
    raw_refresh = create_refresh_token()

    token_row = RefreshToken(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=_hash_token(raw_refresh),
        expires_at=_utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(token_row)
    db.commit()

    return TokenResponse(access_token=access_token, refresh_token=raw_refresh)


def rotate_refresh_token(db: Session, raw_token: str) -> TokenResponse:
    """Verilen refresh token'i iptal eder ve yeni bir cift dondurur (rotation).
    Cok kez kullanma denemesi (calinmis token) revoked_at kontroluyle engellenir."""
    token_hash = _hash_token(raw_token)
    token_row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()

    if (
        token_row is None
        or token_row.revoked_at is not None
        or token_row.expires_at < _utcnow()
    ):
        raise AuthenticationError("Gecersiz ya da suresi dolmus refresh token")

    user = db.get(User, token_row.user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Kullanici bulunamadi ya da devre disi")

    token_row.revoked_at = _utcnow()
    db.add(token_row)

    return issue_tokens(db, user)


def revoke_refresh_token(db: Session, raw_token: str) -> None:
    token_hash = _hash_token(raw_token)
    token_row = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if token_row is not None and token_row.revoked_at is None:
        token_row.revoked_at = _utcnow()
        db.add(token_row)
        db.commit()


def create_user(db: Session, email: str, password: str, display_name: str, role: str) -> User:
    normalized_email = email.lower().strip()
    if db.query(User).filter(User.email == normalized_email).first() is not None:
        raise ConflictError("Bu e-posta adresiyle zaten bir kullanici var")

    user = User(
        id=uuid.uuid4(),
        email=normalized_email,
        password_hash=hash_password(password),
        display_name=display_name,
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def list_users(db: Session) -> list[User]:
    return db.query(User).order_by(User.created_at).all()


def update_user(db: Session, user_id: uuid.UUID, display_name: str | None, role: str | None, is_active: bool | None) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise NotFoundError("Kullanici bulunamadi")

    if display_name is not None:
        user.display_name = display_name
    if role is not None:
        user.role = role
    if is_active is not None:
        user.is_active = is_active

    db.add(user)
    db.commit()
    db.refresh(user)
    return user
