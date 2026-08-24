# auth FastAPI bagimliliklari - eskiden app/routers/photos.py'deki
# verify_service_key() fonksiyonunun ve SERVICE_KEY'in yerini alir.

from uuid import UUID

import jwt
from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationError, ForbiddenError
from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import get_db


def get_current_user(
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("Eksik ya da hatali Authorization basligi")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        raise AuthenticationError("Gecersiz ya da suresi dolmus token")

    if payload.get("type") != "access":
        raise AuthenticationError("Gecersiz token turu")

    try:
        user_id = UUID(payload["sub"])
    except (KeyError, ValueError, TypeError):
        raise AuthenticationError("Gecersiz token govdesi")

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Kullanici bulunamadi ya da devre disi")

    return user


def require_role(*roles: str):
    """Depends(require_role("admin", "editor")) gibi kullanilir."""

    def _dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise ForbiddenError("Bu islem icin yetkiniz yok")
        return user

    return _dependency
