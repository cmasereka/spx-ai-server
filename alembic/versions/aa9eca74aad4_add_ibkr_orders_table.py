"""Add ibkr_orders table and paper_trading_runs table

Revision ID: aa9eca74aad4
Revises: c392c611399e
Create Date: 2026-02-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = 'aa9eca74aad4'
down_revision: Union[str, None] = 'c392c611399e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create paper_trading_runs table (if not already present from a manual migration)
    op.create_table(
        'paper_trading_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', sa.String(50), nullable=False, unique=True, index=True),
        sa.Column('mode', sa.String(20), nullable=False),
        sa.Column('trade_date', sa.Date(), nullable=False),
        sa.Column('strategy_type', sa.String(20), nullable=False),
        sa.Column('parameters', JSONB(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('successful_trades', sa.Integer(), nullable=True),
        sa.Column('total_pnl', sa.Float(), nullable=True),
    )

    # Create ibkr_orders table
    op.create_table(
        'ibkr_orders',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('order_id', sa.String(50), nullable=False, unique=True, index=True),
        sa.Column('session_id', sa.String(50), nullable=False, index=True),
        sa.Column('symbol', sa.String(20), nullable=False, server_default='SPXW'),
        sa.Column('strategy_type', sa.String(30), nullable=False),
        sa.Column('is_entry', sa.Boolean(), nullable=False),
        sa.Column('limit_price', sa.Float(), nullable=False),
        sa.Column('fill_price', sa.Float(), nullable=False),
        sa.Column('slippage', sa.Float(), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('success', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.String(8), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('broker_data', JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('ibkr_orders')
    op.drop_table('paper_trading_runs')
