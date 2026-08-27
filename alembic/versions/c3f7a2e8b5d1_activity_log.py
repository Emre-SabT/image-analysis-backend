"""activity_log tablosu - kurumsal ortak havuzda TUM kullanicilarin
gorebilecegi islem gunlugu (silme, birlestirme, albume ekleme/cikarma,
yeniden adlandirma, yuz atama, vb.)

Revision ID: c3f7a2e8b5d1
Revises: b2e6c9a4f1d7
Create Date: 2026-08-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c3f7a2e8b5d1'
down_revision: Union[str, Sequence[str], None] = 'b2e6c9a4f1d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'activity_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('actor_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('target_kind', sa.String(), nullable=False),
        sa.Column('target_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('target_label', sa.String(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        # SET NULL (photos.uploaded_by_user_id/persons.created_by_user_id ile
        # AYNI ilke, bkz. migration b1c4d6e8f0a2) - kullanici kalici silinse
        # bile gecmis islem kaydi KAYBOLMAZ.
        sa.ForeignKeyConstraint(['actor_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    # En sik sorgu: en yeniden eskiye tam liste (GET /activity) - tek kolonlu
    # index yeterli, target_id/kind'a gore filtreleme su an YOK.
    op.create_index('ix_activity_log_created_at', 'activity_log', ['created_at'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_activity_log_created_at', table_name='activity_log')
    op.drop_table('activity_log')
