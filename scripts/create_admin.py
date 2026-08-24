r"""
Ilk admin hesabini olusturur - coklu kullanicili sisteme gecisle birlikte
acik self-register KAPATILDI (kullanicinin karari: sadece admin yeni
kullanici olusturabilir), bu yuzden ilk admin hesabinin bir sekilde elle
olusturulmasi gerekiyor. Sonraki kullanicilar POST /users (admin-only) ile
olusturulur - bu script sadece bootstrap icin.

Kalici, tekrar kullanilabilir bir arac (scripts/backup_system.py gibi):
her yeni ortam kurulumunda (deploy, test DB, yerel gelistirme) tekrar
gerekecek, tek seferlik bir inceleme scripti degil.

Kullanim:
    python scripts/create_admin.py --email admin@ornek.com --password guclu-bir-parola --display-name "Ad Soyad"

Zaten o e-postayla bir kullanici varsa hata verir (parolasini SIFIRLAMAZ -
kasitli olarak yanlislikla mevcut bir hesabin uzerine yazmayi engeller).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, EmailStr, ValidationError

from app.core.exceptions import ConflictError
from app.db.session import SessionLocal
from app.services import auth_service


class _EmailCheck(BaseModel):
    email: EmailStr


def main() -> None:
    parser = argparse.ArgumentParser(description="Ilk admin kullanicisini olusturur")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--display-name", required=True)
    args = parser.parse_args()

    # POST /auth/login gövdesi de EmailStr kullanıyor - burada aynı kontrolü
    # yapmazsak, .local/.test gibi rezerve bir TLD ile "başarıyla" oluşturulan
    # bir hesaba API üzerinden asla giriş yapılamaz.
    try:
        _EmailCheck(email=args.email)
    except ValidationError as e:
        print(f"HATA: gecersiz e-posta adresi - {e.errors()[0]['msg']}", file=sys.stderr)
        sys.exit(1)

    if len(args.password) < 8:
        print("HATA: parola en az 8 karakter olmali", file=sys.stderr)
        sys.exit(1)

    db = SessionLocal()
    try:
        user = auth_service.create_user(
            db, args.email, args.password, args.display_name, role="admin"
        )
    except ConflictError as e:
        print(f"HATA: {e.message}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()

    print(f"Admin kullanici olusturuldu: {user.email} (id={user.id})")


if __name__ == "__main__":
    main()
