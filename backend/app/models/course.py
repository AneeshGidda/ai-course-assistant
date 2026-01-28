"""
Course / Lecture / Chunk database models.
"""
from sqlalchemy import Column, String, Integer, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
import uuid

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except ImportError:
    # Fallback for type checking
    Vector = None

from app.config import EMBEDDING_DIMENSION

Base = declarative_base()


class ChunkModel(Base):
    """
    Database model for storing chunks with embeddings.
    
    Supports vector similarity search using pgvector.
    """
    __tablename__ = "chunks"
    
    # Primary key
    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Source file information
    file_path = Column(String, nullable=False, index=True)
    source_type = Column(String, nullable=False, index=True)
    
    # Chunk content
    text = Column(Text, nullable=False)
    
    # Locator information (stored as JSON for flexibility)
    locator = Column(JSON, nullable=False)
    
    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    heading = Column(String, nullable=True)
    
    # Vector embedding for similarity search
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=True)
    
    # Timestamps for tracking
    created_at = Column(String, nullable=False)  # ISO format string
    updated_at = Column(String, nullable=False)  # ISO format string
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_file_path_source_type', 'file_path', 'source_type'),
        Index('idx_source_type', 'source_type'),
    )
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "source_type": self.source_type,
            "text": self.text,
            "locator": self.locator,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "heading": self.heading,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
