"""persons/clusters.centroid (PR-A: centroid PG'de otoriter olacak - sema)

TEK AMACLI MIGRATION: sadece kolon ekler, hicbir mevcut kolona/kisita/
indekse DOKUNMAZ, kolayca geri alinabilir. Bu asamada HICBIR OKUYUCU bu
kolonlari kullanmiyor (PR-C'ye kadar) - Qdrant identity_pool hala tek
otoriter kaynak, bu sadece PARALEL bir kopyanin altyapisini kurar.

NEDEN AYRI TABLO DEGIL (identity_centroids gibi), DOGRUDAN KOLON:
  - FK'siz ayri bir tablo, Person/Cluster silindiginde ELLE temizlenmesi
    gereken UCUNCU bir kayit demek - bu oturumda TAM OLARAK bu sinif iki
    bug bulduk (FK ihlali - task_ef05e6fa; "anna" - merge sonrasi
    face_count senkron kalmiyor). Dogrudan kolon, Person/Cluster silinince
    centroid'in de OTOMATIK gitmesini garanti eder - ayri temizlik adimi
    YOK.
  - TOAST: 512x4=2KB'lik BYTEA, Postgres tarafindan otomatik ayri
    depolanir (out-of-line) - centroid SECMEYEN sorgular (list_persons()
    vb.) bu kolonun varliğindan HIC etkilenmez.
  - UNION ALL maliyeti (arama tarafinda persons+clusters birlestirme)
    onceden abartilmisti - iki onceden-filtrelenmis, indeksli sorgunun
    UNION ALL'i (DISTINCT DEGIL) gercek bir maliyet degil.

NEDEN AYRI BIR member_count KOLONU YOK: persons.face_count / clusters.size
ZATEN var olan sayaclar - centroid ile AYNI satirda, AYNI SELECT FOR UPDATE
kilidi altinda guncellenecekler (PR-C). Ucuncu bir kopya yaratmiyoruz.

clusters.centroid_updated_at ZATEN VAR (bkz. models.py) - sadece
persons.centroid_updated_at ekleniyor, sema simetrik hale geliyor.

Revision ID: a7b3c9d1e5f2
Revises: f1a2b3c4d5e6
Create Date: 2026-08-21 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a7b3c9d1e5f2'
down_revision: Union[str, Sequence[str], None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('persons', sa.Column('centroid', sa.LargeBinary(), nullable=True))
    op.add_column('persons', sa.Column('centroid_updated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('clusters', sa.Column('centroid', sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('clusters', 'centroid')
    op.drop_column('persons', 'centroid_updated_at')
    op.drop_column('persons', 'centroid')
