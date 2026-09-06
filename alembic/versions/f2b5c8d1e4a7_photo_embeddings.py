"""photo_embeddings tablosu - semantik arama vektor indeksleme durumu.

Vektorun KENDISI Qdrant 'photo_semantic' koleksiyonunda tutulur
(point_id = photo_id); bu tablo yalnizca "hangi foto, hangi model/boyutla,
ne zaman indekslendi" bilgisini tasir - backfill'in "kalanlar" sorgusu ve
embedding provider/model degisiminde yeniden-indeksleme icin. photo_exif
ile ayni desen (ForeignKey ondelete=CASCADE, tek kolonlu PK).

Revision ID: f2b5c8d1e4a7
Revises: e1f4a7b2c5d8
Create Date: 2026-08-27 00:00:01.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f2b5c8d1e4a7'
down_revision: Union[str, Sequence[str], None] = 'e1f4a7b2c5d8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'photo_embeddings',
        sa.Column('photo_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('dim', sa.Integer(), nullable=False),
        sa.Column('indexed_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('photo_id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('photo_embeddings')
