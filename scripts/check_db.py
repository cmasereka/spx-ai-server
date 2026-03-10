"""
Diagnostic script — inspect DB schema and record counts.

Usage:
    python scripts/check_db.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text, inspect as sa_inspect
from src.database.connection import db_manager


def main():
    engine = db_manager.engine
    inspector = sa_inspect(engine)

    print("=" * 60)
    print("DATABASE DIAGNOSTIC")
    print("=" * 60)

    # Alembic version
    with engine.connect() as conn:
        try:
            rows = conn.execute(text("SELECT version_num FROM alembic_version")).fetchall()
            print(f"\nalembic_version: {[r[0] for r in rows]}")
        except Exception as e:
            print(f"\nalembic_version: ERROR — {e}")

    # Tables
    tables = inspector.get_table_names()
    print(f"\nTables: {sorted(tables)}")

    # backtest_runs columns
    if "backtest_runs" in tables:
        cols = [c["name"] for c in inspector.get_columns("backtest_runs")]
        print(f"\nbacktest_runs columns: {cols}")
        has_user_id = "user_id" in cols
        print(f"  → user_id column present: {has_user_id}")
    else:
        print("\nbacktest_runs table: NOT FOUND")

    # paper_trading_runs columns
    if "paper_trading_runs" in tables:
        cols = [c["name"] for c in inspector.get_columns("paper_trading_runs")]
        has_user_id = "user_id" in cols
        print(f"\npaper_trading_runs.user_id present: {has_user_id}")

    # Record counts
    with engine.connect() as conn:
        for table in ["backtest_runs", "trades", "paper_trading_runs", "users", "invitations"]:
            if table in tables:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"\n{table} row count: {count}")

        # Most recent backtest_runs
        if "backtest_runs" in tables:
            print("\nMost recent backtest_runs (last 5):")
            rows = conn.execute(
                text("SELECT backtest_id, status, created_at FROM backtest_runs ORDER BY created_at DESC LIMIT 5")
            ).fetchall()
            if rows:
                for r in rows:
                    print(f"  {r[0]}  status={r[1]}  created={r[2]}")
            else:
                print("  (none)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
