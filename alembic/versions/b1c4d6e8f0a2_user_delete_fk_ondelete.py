"""kullanici kalici silme icin FK ondelete davranislari

Frontend Ayarlar > Kullanicilar sayfasindaki "devre disi birakma"nin
yaninda GERCEK bir DELETE /users/{id} eklenecek (bkz.
BACKEND_IHTIYACLARI.md #8). Bu, ilgili tablolardaki users.id FK'lerinin
varsayilan davranisi (Postgres NO ACTION/RESTRICT) yuzunden bugun HER
ZAMAN basarisiz olurdu - bir kullanicinin tek bir fotografi/kisisi/
refresh token'i bile varsa silme patlar.

Karar tablo bazinda FARKLI (BACKEND_IHTIYACLARI.md #8'deki "sessiz CASCADE
YAPILMAMALI" uyarisina uyularak):
  - refresh_tokens, user_job_counters: SET NULL ANLAMSIZ (bu satirlar
    KULLANICIYA AIT, baska hicbir sey icin anlami yok) -> CASCADE.
  - photos.uploaded_by_user_id, persons.created_by_user_id,
    cluster_constraints.created_by_user_id: fotograf/kisi/kisit
    KULLANICI SILINSE DE ANLAMLI KALIR (zaten NULL'a izin veriyorlardi,
    coklu-kullanici gecisinden once yuklenen 446 foto icin de boyle) ->
    SET NULL, veri KAYBOLMAZ.
  - jobs.user_id: NOT NULL VE bilerek DEGISTIRILMIYOR - is kuyrugu adil
    siralamasi (sequence_in_user) kullaniciya bagli. Bu FK RESTRICT
    kaliyor; auth_service.delete_user is gecmisi olan bir kullaniciyi
    servis katmaninda ACIKCA reddedip "once devre disi birakin" der -
    kullanicinin audit izi sessizce silinmez.

Revision ID: b1c4d6e8f0a2
Revises: a7b3c9d1e5f2
Create Date: 2026-08-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b1c4d6e8f0a2'
down_revision: Union[str, Sequence[str], None] = 'a7b3c9d1e5f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint('refresh_tokens_user_id_fkey', 'refresh_tokens', type_='foreignkey')
    op.create_foreign_key(
        'refresh_tokens_user_id_fkey', 'refresh_tokens', 'users', ['user_id'], ['id'], ondelete='CASCADE'
    )

    op.drop_constraint('user_job_counters_user_id_fkey', 'user_job_counters', type_='foreignkey')
    op.create_foreign_key(
        'user_job_counters_user_id_fkey', 'user_job_counters', 'users', ['user_id'], ['id'], ondelete='CASCADE'
    )

    op.drop_constraint('fk_photos_uploaded_by_user_id', 'photos', type_='foreignkey')
    op.create_foreign_key(
        'fk_photos_uploaded_by_user_id', 'photos', 'users', ['uploaded_by_user_id'], ['id'], ondelete='SET NULL'
    )

    op.drop_constraint('fk_persons_created_by_user_id', 'persons', type_='foreignkey')
    op.create_foreign_key(
        'fk_persons_created_by_user_id', 'persons', 'users', ['created_by_user_id'], ['id'], ondelete='SET NULL'
    )

    op.drop_constraint('fk_cluster_constraints_created_by_user_id', 'cluster_constraints', type_='foreignkey')
    op.create_foreign_key(
        'fk_cluster_constraints_created_by_user_id', 'cluster_constraints', 'users',
        ['created_by_user_id'], ['id'], ondelete='SET NULL'
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('fk_cluster_constraints_created_by_user_id', 'cluster_constraints', type_='foreignkey')
    op.create_foreign_key(
        'fk_cluster_constraints_created_by_user_id', 'cluster_constraints', 'users',
        ['created_by_user_id'], ['id']
    )

    op.drop_constraint('fk_persons_created_by_user_id', 'persons', type_='foreignkey')
    op.create_foreign_key(
        'fk_persons_created_by_user_id', 'persons', 'users', ['created_by_user_id'], ['id']
    )

    op.drop_constraint('fk_photos_uploaded_by_user_id', 'photos', type_='foreignkey')
    op.create_foreign_key(
        'fk_photos_uploaded_by_user_id', 'photos', 'users', ['uploaded_by_user_id'], ['id']
    )

    op.drop_constraint('user_job_counters_user_id_fkey', 'user_job_counters', type_='foreignkey')
    op.create_foreign_key(
        'user_job_counters_user_id_fkey', 'user_job_counters', 'users', ['user_id'], ['id']
    )

    op.drop_constraint('refresh_tokens_user_id_fkey', 'refresh_tokens', type_='foreignkey')
    op.create_foreign_key(
        'refresh_tokens_user_id_fkey', 'refresh_tokens', 'users', ['user_id'], ['id']
    )
