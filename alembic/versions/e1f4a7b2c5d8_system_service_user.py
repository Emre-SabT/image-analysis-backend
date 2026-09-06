"""system servis hesabi - otomatik (yukleyicisi bilinmeyen) semantic_index
job'lari icin sabit user_id.

Bu hesap NORMAL login/auth akislarinda KULLANILAMAZ:
  - is_active=False; get_current_user (app/core/dependencies.py) ve
    auth_service.authenticate / rotate_refresh_token bunu reddeder.
  - password_hash='!' bilincli olarak gecersiz bir bcrypt formati -
    verify_password ValueError'i yakalayip False doner (app/core/security.py).

Revision ID: e1f4a7b2c5d8
Revises: d4a9c6b2e3f8
Create Date: 2026-08-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e1f4a7b2c5d8'
down_revision: Union[str, Sequence[str], None] = 'd4a9c6b2e3f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ON CONFLICT DO NOTHING: hem id (PK) hem email (UNIQUE) icin idempotent -
    # migration tekrar calissa ya da hesap elle olusturulmus olsa da patlamaz.
    # created_at acikca now() ile set edilir: models.User.created_at yalnizca
    # Python-side default'a sahip, ham SQL insert'inde tetiklenmez.
    op.execute(
        """
        INSERT INTO users (id, email, password_hash, display_name, role, is_active, created_at)
        VALUES (
            '00000000-0000-0000-0000-000000000001',
            'system@photoai.internal',
            '!',
            'Sistem (otomatik analiz)',
            'viewer',
            false,
            now()
        )
        ON CONFLICT DO NOTHING
        """
    )


def downgrade() -> None:
    """Downgrade schema.

    DIKKAT: bu hesap adina en az bir job (semantic_index) kuyruga girdiyse
    jobs.user_id FK'si (NOT NULL, ON DELETE tanimsiz = RESTRICT) bu DELETE'i
    REDDEDER. Downgrade yalnizca hic semantik job uretilmemisken calisir.
    """
    op.execute("DELETE FROM users WHERE id = '00000000-0000-0000-0000-000000000001'")
