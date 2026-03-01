#!/usr/bin/env python3
"""
Database initialization script for SPX AI Trading Platform
Creates database, runs migrations, and sets up initial data
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.connection import db_manager, init_database
from config.settings import DATABASE_URL, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    """Create the PostgreSQL database if it doesn't exist"""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    try:
        # Connect to PostgreSQL server (not specific database)
        # Connect to postgres database instead of user database
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=os.getenv("POSTGRES_PASSWORD", ""),
            database="postgres",  # Connect to postgres system database
            gssencmode="disable"  # Disable GSSAPI
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (POSTGRES_DB,))
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f'CREATE DATABASE "{POSTGRES_DB}"')
            logger.info(f"Created database: {POSTGRES_DB}")
        else:
            logger.info(f"Database {POSTGRES_DB} already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def run_migrations():
    """Run Alembic migrations"""
    import subprocess
    
    try:
        # Generate initial migration if none exist
        result = subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", "Initial migration"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning(f"Migration generation output: {result.stderr}")
        
        # Run migrations
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Migrations completed successfully")
            return True
        else:
            logger.error(f"Migration failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        return False

def test_database():
    """Test database connection and basic operations"""
    try:
        if not db_manager.test_connection():
            logger.error("Database connection test failed")
            return False
        
        # Get database info
        info = db_manager.get_database_info()
        logger.info(f"Connected to: {info}")
        
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    logger.info(f"Database URL: {DATABASE_URL}")
    
    # Step 1: Create database
    logger.info("Step 1: Creating database...")
    if not create_database():
        logger.error("Failed to create database")
        return False
    
    # Step 2: Test connection
    logger.info("Step 2: Testing database connection...")
    if not test_database():
        logger.error("Database connection test failed")
        return False
    
    # Step 3: Run migrations
    logger.info("Step 3: Running migrations...")
    if not run_migrations():
        logger.error("Failed to run migrations")
        return False
    
    # Step 4: Initialize database (create tables)
    logger.info("Step 4: Initializing database tables...")
    if not init_database():
        logger.error("Failed to initialize database")
        return False
    
    logger.info("Database initialization completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)