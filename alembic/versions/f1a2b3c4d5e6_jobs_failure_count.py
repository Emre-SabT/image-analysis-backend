"""jobs.failure_count (PR-1: attempts / gercek-hata sayaci ayrimi)

TEK AMACLI MIGRATION: yalnizca jobs tablosuna failure_count kolonu ekler.
Baska hicbir kolona/kisita/indekse DOKUNMAZ - kolayca geri alinabilir.

NEDEN GEREKLI (bkz. jobs_repository.py basindaki TODO ve is kuyrugu
eszamanlilik analizi): `attempts` kolonu su ana kadar IKI FARKLI anlami
birden tasiyordu:
  1. "kac kez claim edildi" (claim_next HER claim'de - ilk claim VEYA
     reaper sonrasi reclaim - +1 yapiyor)
  2. "kac kez GERCEKTEN denenip basarisiz oldu" (max_attempts karsilastirmasi
     fail()'de bu degeri kullaniyordu)

Bu ikisi CATISIYOR: bir is, worker canliyken yanlislikla reap edilirse
(ornegin OS uyku modu, DB baglanti kesintisi - bkz. analiz) `attempts`
gercek bir hata olmadan artar. Boyle bir is 3 kez yanlis-pozitif reap
yasayip 4. denemede GERCEK (ve tek) bir hataya ugrarsa, eski mantikla
`attempts(4) >= max_attempts(3)` oldugu icin dogrudan kalici 'failed'e
duserdi - halbuki bu onun ILK gercek denemesiydi. Bu, koddaki mevcut,
advisory-lock isinden BAGIMSIZ bir bug'dir (bkz. tests/test_jobs_failure_count.py).

failure_count SADECE fail()'in gercekten cagrildigi (yani handler'in
GERCEKTEN calisip basarisiz oldugu) durumlarda artar; requeue_lock_conflict
(PR-2) gibi "gercek bir deneme sayilmayan" yeniden kuyruklamalar buna
DOKUNMAZ. max_attempts karsilastirmasi artik bu kolon uzerinden yapilir;
`attempts` kolonu "kac kez claim edildi" anlaminda, gozlemlenebilirlik
icin oldugu gibi kaliyor.

Revision ID: f1a2b3c4d5e6
Revises: e8b4d2f1a6c3
Create Date: 2026-08-21 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'e8b4d2f1a6c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'jobs',
        sa.Column('failure_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('jobs', 'failure_count')
