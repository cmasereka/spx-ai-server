"""Add auth tables: users, invitations, user_broker_configs; add user_id to existing tables

Revision ID: b1c2d3e4f5a6
Revises: 20ec909bb215
Create Date: 2026-03-05 00:00:00.000000
"""
from typing import Sequence, Union
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op


revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, None] = '20ec909bb215'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('full_name', sa.String(100), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('role', sa.String(10), nullable=False, server_default='user'),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending_approval'),
        sa.Column('invited_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    op.create_table(
        'invitations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('code', sa.String(64), unique=True, nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('note', sa.String(200), nullable=True),
        sa.Column('is_used', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('used_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_invitations_code', 'invitations', ['code'], unique=True)

    op.create_table(
        'user_broker_configs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('broker_type', sa.String(20), nullable=False),
        sa.Column('label', sa.String(100), nullable=True),
        sa.Column('account_number', sa.String(50), nullable=False),
        sa.Column('is_paper', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('encrypted_credentials', sa.Text(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending_approval'),
        sa.Column('approved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_user_broker_configs_user_id', 'user_broker_configs', ['user_id'])

    op.add_column('backtest_runs',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_backtest_runs_user_id', 'backtest_runs', ['user_id'])

    op.add_column('paper_trading_runs',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index('ix_paper_trading_runs_user_id', 'paper_trading_runs', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_paper_trading_runs_user_id', 'paper_trading_runs')
    op.drop_column('paper_trading_runs', 'user_id')
    op.drop_index('ix_backtest_runs_user_id', 'backtest_runs')
    op.drop_column('backtest_runs', 'user_id')
    op.drop_table('user_broker_configs')
    op.drop_table('invitations')
    op.drop_table('users')
