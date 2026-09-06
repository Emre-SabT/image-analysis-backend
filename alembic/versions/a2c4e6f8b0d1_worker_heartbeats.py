"""worker_heartbeats tablosu - worker sureclerinin canliligini gozlemek.

`jobs` tablosu "bir is kimde" sorusunu yanitliyor ama BOSTAKI bir worker
oradan gorunmez (kilitli satiri yok). Bu tablo her worker surecinin
periyodik olarak yazdigi bir "hayattayim" kaydidir: `/health` (Sistem
durumu) ve `/jobs/queue-status` (Genel Bakis) bunu okuyup her is tipi
icin "worker calisiyor mu" bilgisini gosterir.

VERI DEPOSU DEGIL, koordinasyon/gozlem tablosu - `jobs` ile ayni desen.
Zaman kolonu TIMESTAMPTZ (yine `jobs` gibi): now() karsilastirmalari var,
timezone-aware olmak burada gercek bir dogruluk meselesi.

`worker_id` worker'in kendi atadigi kimlik ({hostname}-{pid}); surec
yeniden basladiginda DEGISIR, bu yuzden eski satirlar birikebilir -
worker reaper dongusu `prune_stale` ile cok eski kayitlari siler.

Revision ID: a2c4e6f8b0d1
Revises: f2b5c8d1e4a7
Create Date: 2026-08-31 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a2c4e6f8b0d1'
down_revision: Union[str, Sequence[str], None] = 'f2b5c8d1e4a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'worker_heartbeats',
        sa.Column('worker_id', sa.Text(), nullable=False),
        # Virgulle ayrilmis JOB_TYPES (worker genelde tek tip tuketir ama
        # coklu tip destekleniyor - bkz. worker/main.parse_job_types).
        sa.Column('job_types', sa.Text(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('worker_id'),
    )
    # Canlilik sorgusu last_seen'e gore filtreler/siralar.
    op.create_index(
        'ix_worker_heartbeats_last_seen', 'worker_heartbeats', ['last_seen']
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_worker_heartbeats_last_seen', table_name='worker_heartbeats')
    op.drop_table('worker_heartbeats')
