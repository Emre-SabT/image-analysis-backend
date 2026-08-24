# parola hash'leme + JWT access token / opak refresh token uretimi
#
# passlib KULLANILMIYOR: passlib[bcrypt] ile guncel bcrypt (>=4.1) surumleri
# arasinda bilinen bir uyumsuzluk var ("module 'bcrypt' has no attribute
# '__about__'"). Dogrudan bcrypt kutuphanesi kullaniliyor - daha az
# bagimlilik, daha az surum catismasi riski.

import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt

from app.core.settings import settings

ACCESS_TOKEN_TYPE = "access"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        # bozuk/gecersiz hash formati - guvenli tarafta kal
        return False


def create_access_token(user_id: UUID, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": ACCESS_TOKEN_TYPE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Gecerli degilse jwt.PyJWTError (ExpiredSignatureError dahil) fırlatır."""
    return jwt.decode(
        token,
        settings.JWT_SECRET,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp", "sub", "type"]},
    )


def create_refresh_token() -> str:
    """Opak, yuksek-entropili token - JWT DEGIL. Login ve kullanici olusturma
    akislarinin ikisi de ayni formati kullanir (referans projedeki gibi iki
    farkli format - biri imzali JWT, digeri opak - arasinda tutarsizlik
    yaratmiyoruz)."""
    return secrets.token_urlsafe(64)
