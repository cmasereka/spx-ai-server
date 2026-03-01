#!/usr/bin/env python3
"""
Database management utilities for SPX AI Trading Platform
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.connection import db_manager
from config.settings import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test database connection"""
    logger.info("Testing database connection...")
    if db_manager.test_connection():
        info = db_manager.get_database_info()
        logger.info(f"Connection successful: {info}")
        return True
    else:
        logger.error("Connection failed")
        return False

def create_tables():
    """Create all database tables"""
    logger.info("Creating database tables...")
    try:
        db_manager.create_tables()
        logger.info("Tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

def drop_tables():
    """Drop all database tables"""
    logger.info("Dropping database tables...")
    try:
        db_manager.drop_tables()
        logger.info("Tables dropped successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        return False

def reset_database():
    """Reset database (drop and recreate tables)"""
    logger.info("Resetting database...")
    if drop_tables() and create_tables():
        logger.info("Database reset completed")
        return True
    else:
        logger.error("Database reset failed")
        return False

def show_info():
    """Show database information"""
    logger.info(f"Database URL: {DATABASE_URL}")
    if db_manager.test_connection():
        info = db_manager.get_database_info()
        print(f"Database: {info.get('database', 'Unknown')}")
        print(f"Version: {info.get('version', 'Unknown')}")
        print(f"URL: {info.get('url', 'Unknown')}")
    else:
        print("Cannot connect to database")

def main():
    parser = argparse.ArgumentParser(description="Database management utilities")
    parser.add_argument("command", choices=["test", "create", "drop", "reset", "info"],
                       help="Command to execute")
    
    args = parser.parse_args()
    
    commands = {
        "test": test_connection,
        "create": create_tables,
        "drop": drop_tables,
        "reset": reset_database,
        "info": show_info
    }
    
    success = commands[args.command]()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()