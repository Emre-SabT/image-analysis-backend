"""is kuyrugu - jobs + user_job_counters (PostgreSQL, FOR UPDATE SKIP LOCKED)

VLM analizi ve yuz hatti artik surec-ici ThreadPoolExecutor yerine kalici
bir kuyruk uzerinden yurutuluyor: surec yeniden baslarsa is kaybolmuyor,
retry var, durum gozlemlenebiliyor (job_id).

Redis/Celery YERINE PostgreSQL secildi: urun on-premise kuruluyor ve
PostgreSQL zaten kurulum listesinde; Redis musteri sahasinda ayri bir
kurulum/izleme/yedekleme yuku demek olurdu.

INDEKS TASARIMI (EXPLAIN ANALYZE ile 200.000 satirda dogrulandi):

  ix_jobs_claim (type, priority DESC, sequence_in_user, created_at)
                WHERE status = 'queued'

  - `type` esitlik kosulu oldugu icin BASA alindi; ardindan gelen kolonlar
    claim_next'in ORDER BY'i ile BIREBIR ayni sirada oldugundan Postgres
    siralamayi indeksten okur (ayri bir Sort dugumu olusmaz) ve LIMIT 1
    ilk satirda durur.
  - KISMI indeks (WHERE status='queued'): tablo done/failed satirlariyla
    sinirsiz buyur ama claim_next yalnizca queued satirlara bakar. Kismi
    indeks tabloyla birlikte buyumez - kuyruk tablolarinda en kritik karar.
    (Olcum: tablo 27 MB iken bu indeks 2.400 kB.)
  - `run_after` BILINCLI olarak indekste DEGIL: siralamanin ortasina bir
    esitsizlik koymak sonraki kolonlarin ORDER BY icin kullanilmasini
    engellerdi. Filtre olarak uygulanmasi olcumde ek maliyet uretmedi.

  Olculen: Index Scan using ix_jobs_claim -> 5 buffer / 0,036 ms
           (karsilastirma icin `type = ANY(...)` yolu: 2.449 buffer / 9,582 ms)

KABUL EDILEN SINIRLAMALAR:
  1. priority starvation: ORDER BY priority DESC, bulk islerin (priority=0)
     suresiz ertelenmesine yol acabilir. v1'de AGING YOK - kabul edilmis
     sinirlama; olcek buyurse bekleme suresine gore priority artirma
     (aging) eklenmeli.
  2. sequence_in_user hic sifirlanmaz: sayac kullanici omru boyunca monoton
     artar, dolayisiyla YENI bir kullanici dusuk sequence'la siraya girip
     eski/yogun bir kullanicinin onune gecer. v1 icin kabul edilebilir;
     gercek adalet gerekirse pencere bazli sifirlama veya sanal-zaman (WFQ).

Revision ID: d5a3f1c7b9e2
Revises: c1e9a2f6b3d4
Create Date: 2026-08-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd5a3f1c7b9e2'
down_revision: Union[str, Sequence[str], None] = 'c1e9a2f6b3d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'user_job_counters',
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('next_sequence', sa.BigInteger(), nullable=False,
                  server_default=sa.text('0')),
    )

    op.create_table(
        'jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('payload', postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column('status', sa.String(length=20), nullable=False,
                  server_default='queued'),
        sa.Column('priority', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id'), nullable=False),
        sa.Column('sequence_in_user', sa.BigInteger(), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('max_attempts', sa.Integer(), nullable=False, server_default=sa.text('3')),
        sa.Column('locked_by', sa.String(length=100), nullable=True),
        sa.Column('locked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('run_after', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    )

    # DESC + kismi indeks: op.create_index bu kombinasyonu temiz ifade
    # edemiyor, ham SQL kullaniliyor.
    op.execute(
        """
        CREATE INDEX ix_jobs_claim
            ON jobs (type, priority DESC, sequence_in_user ASC, created_at ASC)
            WHERE status = 'queued'
        """
    )
    op.execute(
        "CREATE INDEX ix_jobs_reaper ON jobs (locked_at) WHERE status = 'running'"
    )
    op.execute(
        "CREATE INDEX ix_jobs_user_queued ON jobs (user_id) WHERE status = 'queued'"
    )
    op.execute(
        """
        CREATE INDEX ix_jobs_done_recent
            ON jobs (type, finished_at DESC)
            WHERE status = 'done'
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_jobs_done_recent")
    op.execute("DROP INDEX IF EXISTS ix_jobs_user_queued")
    op.execute("DROP INDEX IF EXISTS ix_jobs_reaper")
    op.execute("DROP INDEX IF EXISTS ix_jobs_claim")
    op.drop_table('jobs')
    op.drop_table('user_job_counters')
