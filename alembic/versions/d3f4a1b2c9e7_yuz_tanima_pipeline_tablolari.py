"""yuz tanima pipeline tablolari (faces, clusters, persons, face_suggestions, cluster_constraints)

Revision ID: d3f4a1b2c9e7
Revises: b05ab8dd4e2f
Create Date: 2026-08-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd3f4a1b2c9e7'
down_revision: Union[str, Sequence[str], None] = 'b05ab8dd4e2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'clusters',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('status', sa.String(), server_default='unlabeled', nullable=True),
        sa.Column('size', sa.Integer(), server_default='0', nullable=True),
        sa.Column('centroid_updated_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'persons',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('display_name', sa.String(), nullable=False),
        sa.Column('cluster_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('clusters.id'), nullable=True),
        sa.Column('face_count', sa.Integer(), server_default='0', nullable=True),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'faces',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('photo_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('photos.id', ondelete='CASCADE'), nullable=False),
        sa.Column('bbox', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('landmarks', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('det_confidence', sa.Float(), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('crop_path', sa.String(), nullable=False),
        sa.Column('cluster_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('clusters.id'), nullable=True),
        sa.Column('person_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('persons.id'), nullable=True),
        sa.Column('assigned_by', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_faces_photo_id', 'faces', ['photo_id'])
    op.create_index('ix_faces_cluster_id', 'faces', ['cluster_id'])
    op.create_index('ix_faces_person_id', 'faces', ['person_id'])

    op.create_table(
        'face_suggestions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('face_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('faces.id'), nullable=False),
        sa.Column('person_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('persons.id'), nullable=False),
        sa.Column('similarity', sa.Float(), nullable=False),
        sa.Column('status', sa.String(), server_default='pending', nullable=True),
        sa.Column('resolved_by', sa.String(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_face_suggestions_status', 'face_suggestions', ['status'])

    op.create_table(
        'cluster_constraints',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('face_id_a', postgresql.UUID(as_uuid=True), sa.ForeignKey('faces.id'), nullable=False),
        sa.Column('face_id_b', postgresql.UUID(as_uuid=True), sa.ForeignKey('faces.id'), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('cluster_constraints')
    op.drop_index('ix_face_suggestions_status', table_name='face_suggestions')
    op.drop_table('face_suggestions')
    op.drop_index('ix_faces_person_id', table_name='faces')
    op.drop_index('ix_faces_cluster_id', table_name='faces')
    op.drop_index('ix_faces_photo_id', table_name='faces')
    op.drop_table('faces')
    op.drop_table('persons')
    op.drop_table('clusters')
