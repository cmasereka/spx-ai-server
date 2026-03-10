"""Add phone column to users table

Revision ID: c7d8e9f0a1b2
Revises: b1c2d3e4f5a6
Create Date: 2026-03-05 01:00:00.000000
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = 'c7d8e9f0a1b2'
down_revision: Union[str, None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect as sa_inspect
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    cols = {c['name'] for c in inspector.get_columns('users')}
    if 'phone' not in cols:
        op.add_column('users', sa.Column('phone', sa.String(30), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'phone')
