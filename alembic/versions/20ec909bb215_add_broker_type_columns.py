"""Add broker_type columns to ibkr_orders and paper_trading_runs

Revision ID: 20ec909bb215
Revises: aa9eca74aad4
Create Date: 2026-03-03 00:00:00.000000

Adds broker_type to ibkr_orders (default 'ibkr' for all existing rows) and
a nullable broker_type to paper_trading_runs.  The ibkr_orders table name is
unchanged — 'broker_type' simply identifies which broker submitted each order.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20ec909bb215'
down_revision: Union[str, None] = 'aa9eca74aad4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'ibkr_orders',
        sa.Column('broker_type', sa.String(20), nullable=False, server_default='ibkr'),
    )
    op.add_column(
        'paper_trading_runs',
        sa.Column('broker_type', sa.String(20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('ibkr_orders', 'broker_type')
    op.drop_column('paper_trading_runs', 'broker_type')
