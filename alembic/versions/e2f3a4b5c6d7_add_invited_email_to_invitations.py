"""Add invited_email column to invitations table

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-03-06 00:00:00.000000
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = 'e2f3a4b5c6d7'
down_revision: Union[str, None] = 'd1e2f3a4b5c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect as sa_inspect
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    cols = {c['name'] for c in inspector.get_columns('invitations')}
    if 'invited_email' not in cols:
        op.add_column('invitations', sa.Column('invited_email', sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column('invitations', 'invited_email')
