"""photos.content_hash - dosya tekillestirme (SHA-256)

Ayni dosyanin (bayt bayt ayni icerik) birden fazla yuklenmesini engellemek
icin: photo_service.save_upload() yuklenen dosyanin SHA-256'sini hesaplar,
bu kolonda eslesme bulursa yeni bir photos/faces/photo_analysis kaydi
ACMADAN mevcut fotografi dondurur (bkz. routers/photos.py upload_photo).

Nullable + backfill: bu migration SADECE kolonu ve (duz, UNIQUE OLMAYAN)
index'i ekler. Migration'dan once yuklenmis eski kayitlarin hash'i yoktur
(NULL) - onlar scripts/backfill_content_hash.py ile geriye donuk doldurulur.

UNIQUE DEGIL (bilincli karar): backfill sirasinda, bu ozellik eklenmeden
once yuklenmis 7 grup HALIHAZIRDA yinelenen fotograf bulundu (ayni icerik,
farkli klasor/isimle iki kez yuklenmis - bkz. backfill script ciktisi).
Unique kisit bu satirlarin hash'ini doldururken ilk commit'te patlardi.
Tekillestirme uygulama katmaninda (save_upload, YENI yuklemeler icin)
zaten yapiliyor; DB'deki index sadece sorgu hizini (WHERE content_hash = ...)
saglamak icin, bir bütünlük kisiti degil.

Revision ID: a4b1c9d2e5f6
Revises: e7c1b8a4f2d9
Create Date: 2026-08-07 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a4b1c9d2e5f6'
down_revision: Union[str, Sequence[str], None] = 'e7c1b8a4f2d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('photos', sa.Column('content_hash', sa.String(length=64), nullable=True))
    op.create_index('ix_photos_content_hash', 'photos', ['content_hash'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_photos_content_hash', table_name='photos')
    op.drop_column('photos', 'content_hash')
