"""
Exceptions for RAG operations.
"""
from dataclasses import dataclass


@dataclass
class InsufficientMaterialError(Exception):
    """
    Raised when retrieval quality is too low to generate a reliable answer.
    
    Prevents hallucinations by refusing to answer when there's insufficient
    high-quality context.
    """
    query: str
    top_similarity: float
    high_quality_chunks: int
    total_chunks: int
    message: str
    
    def __str__(self) -> str:
        return self.message
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "error": "insufficient_material",
            "query": self.query,
            "top_similarity": self.top_similarity,
            "high_quality_chunks": self.high_quality_chunks,
            "total_chunks": self.total_chunks,
            "message": self.message,
        }
