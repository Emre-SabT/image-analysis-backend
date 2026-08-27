"""cluster_constraints'e source kolonu eklendi

`type` (must_link/cannot_link) tek basina HANGI eylemden geldigini
ayirt etmiyor - "cannot_link" hem bir birlestirme ONERISININ REDDinden
(reject_merge) hem alakasiz bir "kimlikten ayir" eyleminden (reassign_face
split) yaziliyordu. GET /identities/merge-history bunlari karistirmadan
listeleyebilsin diye `source` eklendi (bkz. app/db/models.py,
app/services/person_service.py). Bu kolon eklenmeden ONCE yazilmis eski
kayitlarin kaynagi GERCEKTEN bilinmiyor - geriye donuk UYDURULMAZ, NULL
kalir (yeni endpoint bu yuzden source'a gore filtreler, eski NULL'lu
kayitlar "kabul/red" listesinde GORUNMEZ).

Revision ID: b2e6c9a4f1d7
Revises: a1f5d8c2b4e6
Create Date: 2026-08-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2e6c9a4f1d7'
down_revision: Union[str, Sequence[str], None] = 'a1f5d8c2b4e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('cluster_constraints', sa.Column('source', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('cluster_constraints', 'source')
