"""Rename paper_trading_runs to live_trading_runs; drop is_paper from user_broker_configs

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-03-23 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = 'f1a2b3c4d5e6'
branch_labels = None
depends_on = None


def upgrade():
    # Rename the table
    op.rename_table('paper_trading_runs', 'live_trading_runs')

    # Drop the is_paper column from user_broker_configs (all accounts are live)
    with op.batch_alter_table('user_broker_configs') as batch_op:
        batch_op.drop_column('is_paper')


def downgrade():
    # Restore is_paper column (default False so existing rows don't break)
    with op.batch_alter_table('user_broker_configs') as batch_op:
        batch_op.add_column(sa.Column('is_paper', sa.Boolean(), nullable=False, server_default='false'))

    op.rename_table('live_trading_runs', 'paper_trading_runs')
