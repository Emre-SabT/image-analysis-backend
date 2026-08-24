"""face_suggestions tablosunu dusur - olu kod temizligi

Hedeflenen tasarimin 3 kademeli (SUGGEST/benzer-kisi/OTOMATIK) esik
bandinin kalintisiydi. Kullanicinin acik istegiyle gercek-zamanli atama
tek esige (AUTO_ASSIGN_THRESHOLD) indirgendiginde bu tablo hic
kullanilmaz hale geldi - canli veride hep 0 kayit vardi, hicbir aktif
kod yolu ne yaziyor ne okuyordu. GET /suggestions uc noktasi ve
person_service.list_suggestions() de bu temizlikle birlikte kaldirildi.

Revision ID: b7c2e8f4a1d3
Revises: a4b1c9d2e5f6
Create Date: 2026-08-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b7c2e8f4a1d3'
down_revision: Union[str, Sequence[str], None] = 'a4b1c9d2e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_table('face_suggestions')


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        'face_suggestions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('face_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('faces.id'), nullable=False),
        sa.Column('person_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('similarity', sa.Float(), nullable=False),
        sa.Column('status', sa.String(), server_default='pending'),
        sa.Column('resolved_by', sa.String(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
