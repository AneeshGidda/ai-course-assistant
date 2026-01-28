"""
Database connection and session management.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from typing import Generator
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from app.config import DATABASE_URL

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # Use NullPool for simplicity, can switch to QueuePool for production
    echo=False,  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database: create database if needed, enable pgvector extension, and create tables.
    
    Should be called once at application startup.
    """
    from app.models.course import Base
    from app.config import DATABASE_URL
    from sqlalchemy import create_engine
    
    # Parse DATABASE_URL to get connection info
    # Format: postgresql://user:password@host:port/dbname
    import re
    url_pattern = r'postgresql://(?:([^:]+):([^@]+)@)?([^:]+):(\d+)/(.+)'
    match = re.match(url_pattern, DATABASE_URL)
    
    if not match:
        raise ValueError(f"Could not parse DATABASE_URL: {DATABASE_URL}")
    
    user, password, host, port, dbname = match.groups()
    user = user or "postgres"
    password = password or ""
    host = host or "localhost"
    port = port or "5432"
    
    # Connect to postgres database (default database) to create our database
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"  # Connect to default postgres database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists (using parameterized query)
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (dbname,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            # Use identifier quoting for database name
            cursor.execute(f'CREATE DATABASE "{dbname}"')
            print(f"Created database: {dbname}")
        else:
            print(f"Database {dbname} already exists")
        
        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        raise Exception(f"Failed to create database: {e}")
    
    # Now connect to our database and enable pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Create all tables
    Base.metadata.create_all(bind=engine)


def reset_db() -> None:
    """
    Drop all tables and recreate them.
    
    WARNING: This will delete all data!
    """
    from app.models.course import Base
    
    Base.metadata.drop_all(bind=engine)
    init_db()
