"""photo_analysis alan sadelestirme (environment_type/people_count/possible_event kaldirildi)

VLM prompt semasi 11 etikete indirildi (bkz. app/ai/dispatcher.py PROMPT) -
environment_type, people_count, possible_event artik uretilmiyor. Bu kolonlar
DB'de kullanilmadigi icin kaldirildi.

Revision ID: a1f5d8c2b4e6
Revises: c2d5e7f9a1b3
Create Date: 2026-08-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1f5d8c2b4e6'
down_revision: Union[str, Sequence[str], None] = 'c2d5e7f9a1b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column('photo_analysis', 'environment_type')
    op.drop_column('photo_analysis', 'people_count')
    op.drop_column('photo_analysis', 'possible_event')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('photo_analysis', sa.Column('possible_event', sa.String(), autoincrement=False, nullable=True))
    op.add_column('photo_analysis', sa.Column('people_count', sa.Integer(), autoincrement=False, nullable=True))
    op.add_column('photo_analysis', sa.Column('environment_type', sa.String(), autoincrement=False, nullable=True))
