"""albumler (BACKEND_IHTIYACLARI.md #1)

Coklu fotografi bir arada gruplama - frontend'in Fotograflar sayfasindaki
coklu secim "Albume ekle" akisi ve yeni /albums, /albums/:id sayfalari
icin. Bir fotograf BIRDEN FAZLA albumde olabilmeli (mockup'ta kisi/foto
basina coklu album ornekleri var) - bu yuzden `Photo.album_id` DEGIL,
ayri bir join tablosu (`album_photos`).

Kullanicinin acik karariyla: mevcut fotograflar hicbir albume otomatik
eklenmez (bos albumler olusturulmaz, sahte veri UYDURULMAZ) - kullanici
GERCEKTEN fotograf ekleyene kadar her albüm bos baslar.

Revision ID: c2d5e7f9a1b3
Revises: b1c4d6e8f0a2
Create Date: 2026-08-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c2d5e7f9a1b3'
down_revision: Union[str, Sequence[str], None] = 'b1c4d6e8f0a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'albums',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        # Kapak fotoğrafı - liste/kart görünümünde thumbnail için. Albüm
        # silinse de fotoğraf silinmez (bkz. album_photos ondelete), bu
        # yüzden burada CASCADE YOK - kapak fotoğrafı silinirse SET NULL
        # (kapaksız albüm gösterilir, hiçbir şey patlamaz).
        sa.Column('cover_photo_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        'fk_albums_created_by_user_id', 'albums', 'users', ['created_by_user_id'], ['id'], ondelete='SET NULL'
    )
    op.create_foreign_key(
        'fk_albums_cover_photo_id', 'albums', 'photos', ['cover_photo_id'], ['id'], ondelete='SET NULL'
    )

    op.create_table(
        'album_photos',
        sa.Column('album_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('photo_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('added_at', sa.DateTime(), nullable=True),
        sa.Column('added_by_user_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        'fk_album_photos_album_id', 'album_photos', 'albums', ['album_id'], ['id'], ondelete='CASCADE'
    )
    op.create_foreign_key(
        'fk_album_photos_photo_id', 'album_photos', 'photos', ['photo_id'], ['id'], ondelete='CASCADE'
    )
    op.create_foreign_key(
        'fk_album_photos_added_by_user_id', 'album_photos', 'users', ['added_by_user_id'], ['id'], ondelete='SET NULL'
    )
    # Bir albümün fotoğraf listesini/sayısını çekerken (GET /albums,
    # GET /albums/{id}/photos) kullanılan asıl erişim yolu.
    op.create_index('ix_album_photos_album_id', 'album_photos', ['album_id'])
    op.create_index('ix_album_photos_photo_id', 'album_photos', ['photo_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_album_photos_photo_id', table_name='album_photos')
    op.drop_index('ix_album_photos_album_id', table_name='album_photos')
    op.drop_table('album_photos')
    op.drop_table('albums')
