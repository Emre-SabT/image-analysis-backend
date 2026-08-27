"""photo_exif tablosu (BACKEND_IHTIYACLARI.md #6) - dosyadan gercekten
okunan kamera/pozlama/GPS/telif meta verisi.

Revision ID: d4a9c6b2e3f8
Revises: c3f7a2e8b5d1
Create Date: 2026-08-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd4a9c6b2e3f8'
down_revision: Union[str, Sequence[str], None] = 'c3f7a2e8b5d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'photo_exif',
        sa.Column('photo_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('camera_make', sa.String(), nullable=True),
        sa.Column('camera_model', sa.String(), nullable=True),
        sa.Column('lens_model', sa.String(), nullable=True),
        sa.Column('aperture', sa.String(), nullable=True),
        sa.Column('shutter_speed', sa.String(), nullable=True),
        sa.Column('iso', sa.Integer(), nullable=True),
        sa.Column('focal_length', sa.String(), nullable=True),
        sa.Column('captured_at', sa.DateTime(), nullable=True),
        sa.Column('gps_latitude', sa.Float(), nullable=True),
        sa.Column('gps_longitude', sa.Float(), nullable=True),
        sa.Column('copyright', sa.String(), nullable=True),
        sa.Column('width_px', sa.Integer(), nullable=True),
        sa.Column('height_px', sa.Integer(), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(['photo_id'], ['photos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('photo_id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('photo_exif')
