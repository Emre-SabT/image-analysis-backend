"""coklu kullanici (JWT auth) - users, refresh_tokens, created_by FK'leri

Paylasilan kurumsal arsiv modeli: coklu kullanici destegi icin gerekli
tablolar (kullanicinin karari: sadece admin yeni kullanici olusturabilir,
3 rol: admin/editor/viewer). Eskiden serbest metin olan Person.created_by
ve ClusterConstraint.created_by artik gercek users.id FK'si
(created_by_user_id) - istemcinin gonderdigi string yerine
get_current_user'dan gelen kimlikten yaziliyor.

Photo tablosuna da uploaded_by_user_id eklendi - bugune kadar hicbir
sahiplik izi yoktu, gercek denetim izi icin gerekli.

Kullanicinin acik karariyla: mevcut 446 foto / 104 kisi kaydinin yeni FK
alanlari NULL kalir (sahte sahiplik uydurulmadi, veri tasima yapilmadi).

Revision ID: c1e9a2f6b3d4
Revises: b7c2e8f4a1d3
Create Date: 2026-08-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c1e9a2f6b3d4'
down_revision: Union[str, Sequence[str], None] = 'b7c2e8f4a1d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('display_name', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False, server_default='viewer'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    op.create_table(
        'refresh_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('token_hash', sa.String(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'], unique=False)
    op.create_index('ix_refresh_tokens_token_hash', 'refresh_tokens', ['token_hash'], unique=True)

    op.add_column('photos', sa.Column('uploaded_by_user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        'fk_photos_uploaded_by_user_id', 'photos', 'users', ['uploaded_by_user_id'], ['id']
    )

    op.add_column('persons', sa.Column('created_by_user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        'fk_persons_created_by_user_id', 'persons', 'users', ['created_by_user_id'], ['id']
    )
    op.drop_column('persons', 'created_by')

    op.add_column('cluster_constraints', sa.Column('created_by_user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        'fk_cluster_constraints_created_by_user_id', 'cluster_constraints', 'users', ['created_by_user_id'], ['id']
    )
    op.drop_column('cluster_constraints', 'created_by')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('cluster_constraints', sa.Column('created_by', sa.String(), nullable=True))
    op.drop_constraint('fk_cluster_constraints_created_by_user_id', 'cluster_constraints', type_='foreignkey')
    op.drop_column('cluster_constraints', 'created_by_user_id')

    op.add_column('persons', sa.Column('created_by', sa.String(), nullable=True))
    op.drop_constraint('fk_persons_created_by_user_id', 'persons', type_='foreignkey')
    op.drop_column('persons', 'created_by_user_id')

    op.drop_constraint('fk_photos_uploaded_by_user_id', 'photos', type_='foreignkey')
    op.drop_column('photos', 'uploaded_by_user_id')

    op.drop_index('ix_refresh_tokens_token_hash', table_name='refresh_tokens')
    op.drop_index('ix_refresh_tokens_user_id', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')

    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
