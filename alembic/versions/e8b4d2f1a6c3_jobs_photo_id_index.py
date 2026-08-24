"""jobs.payload->>'photo_id' expression index (SADECE indeks)

TEK AMACLI MIGRATION: yalnizca ix_jobs_photo_id indeksini ekler.
jobs tablosunun hicbir kolonuna, kisitina veya diger indekslerine
DOKUNMAZ - tamamen ve kolayca geri alinabilir.

Neden gerekli: GET /photos/{id}/status ve GET /photos/status?ids=...
uc noktalari bir fotografin iki isini payload icindeki photo_id ile
buluyor. Bu uclar frontend tarafindan POLLING ile surekli cagriliyor
ve `jobs` tablosu zamanla sinirsiz buyuyor - indekssiz her sorgu
seq scan olurdu.

NOT (Adim 5 dersi baglaminda): buradaki toplu sorgu `= ANY(:ids)`
kullaniyor ama claim_next'teki sorunun aksine burada ORDER BY YOK.
ScalarArrayOpExpr'in siralama garantisini bozmasi yalnizca ORDER BY
ile birlikte sorun olur; duz arama icin sorunsuz (bitmap scan).

Revision ID: e8b4d2f1a6c3
Revises: d5a3f1c7b9e2
Create Date: 2026-08-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e8b4d2f1a6c3'
down_revision: Union[str, Sequence[str], None] = 'd5a3f1c7b9e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "CREATE INDEX ix_jobs_photo_id ON jobs ((payload->>'photo_id'))"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_jobs_photo_id")
