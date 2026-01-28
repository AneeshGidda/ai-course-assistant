#!/usr/bin/env python3
"""
Initialize the database: create tables and enable pgvector extension.

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.db.database import init_db


def main():
    """Initialize the database."""
    try:
        print("Initializing database...")
        print("  - Enabling pgvector extension")
        print("  - Creating tables")
        
        init_db()
        
        print("\nDatabase initialized successfully!")
        print("\nYou can now run ingestion:")
        print("  python scripts/ingest_course.py <course_code>")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
