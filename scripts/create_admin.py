"""
Bootstrap script — creates the initial admin user.

Usage:
    ADMIN_EMAIL=admin@example.com \
    ADMIN_PASSWORD=ChangeMe123! \
    ADMIN_FULL_NAME="Admin User" \
    python scripts/create_admin.py

Skips silently if the email already exists.
"""

import os
import sys

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.database.connection import db_manager
from src.database.models import User
from api.auth import hash_password


def main():
    email = os.environ.get("ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    full_name = os.environ.get("ADMIN_FULL_NAME", "Admin").strip()

    if not email or not password:
        print("ERROR: ADMIN_EMAIL and ADMIN_PASSWORD environment variables must be set.")
        sys.exit(1)

    with db_manager.get_session() as db:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"Admin user already exists: {email}")
            return

        admin = User(
            email=email,
            full_name=full_name,
            hashed_password=hash_password(password),
            role="admin",
            status="approved",
        )
        db.add(admin)
        db.commit()
        print(f"Admin user created: {email}")


if __name__ == "__main__":
    main()
