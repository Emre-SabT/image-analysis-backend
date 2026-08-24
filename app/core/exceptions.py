# auth/kullanici is mantigi icin kucuk bir hata hiyerarsisi
#
# IntelliumAI-Backend referansindaki ServiceError deseninden esinlenilmis,
# sadelestirilmis hali. Mevcut router'lardaki ValueError -> HTTPException
# deseni degismedi; bu hiyerarsi sadece yeni auth/kullanici kodunda kullanilir.

from fastapi import Request
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    status_code: int = 500

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class AuthenticationError(ServiceError):
    """Gecersiz kimlik bilgileri / gecersiz ya da suresi dolmus token."""
    status_code = 401


class ForbiddenError(ServiceError):
    """Kimlik dogrulandi ama rol bu islem icin yetersiz."""
    status_code = 403


class NotFoundError(ServiceError):
    status_code = 404


class ConflictError(ServiceError):
    """Ornegin: email zaten kayitli."""
    status_code = 409


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.message},
    )
