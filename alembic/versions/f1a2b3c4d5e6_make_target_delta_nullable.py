"""Make target_delta nullable in backtest_runs

target_delta was removed from the ORM model when delta/Black-Scholes strike
selection was dropped, but the column remained NOT NULL with no default in the
database.  Every new INSERT into backtest_runs therefore raised
NotNullViolation, which was silently swallowed — causing all new backtest
records to be lost after that change was deployed.

Fix: make the column nullable so existing rows keep their value and new rows
(which no longer include target_delta) insert NULL without error.

Revision ID: f1a2b3c4d5e6
Revises: e2f3a4b5c6d7
Create Date: 2026-03-10 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e2f3a4b5c6d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect as sa_inspect
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    cols = {c['name']: c for c in inspector.get_columns('backtest_runs')}
    if 'target_delta' in cols and not cols['target_delta']['nullable']:
        op.alter_column('backtest_runs', 'target_delta', nullable=True)


def downgrade() -> None:
    # Fill NULLs with 0.0 before re-applying NOT NULL
    op.execute("UPDATE backtest_runs SET target_delta = 0.0 WHERE target_delta IS NULL")
    op.alter_column('backtest_runs', 'target_delta', nullable=False)
