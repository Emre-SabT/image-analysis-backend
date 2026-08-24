"""faces.is_background - arka plan yuzu isareti (Bolum 8.1, %0.1 alan esigi)

Revision ID: e7c1b8a4f2d9
Revises: d3f4a1b2c9e7
Create Date: 2026-08-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e7c1b8a4f2d9'
down_revision: Union[str, Sequence[str], None] = 'd3f4a1b2c9e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'faces',
        sa.Column('is_background', sa.Boolean(), nullable=False, server_default=sa.text('false')),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('faces', 'is_background')
