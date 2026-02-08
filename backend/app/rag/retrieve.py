"""
Vector search and retrieval for user queries.

Provides functions to retrieve relevant chunks using vector similarity search
with support for filtering by course and source type.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from .vector_store import VectorStore
from .schemas import Chunk, ChunkLocator, SourceType
from app.models.course import ChunkModel


@dataclass
class RetrievalResult:
    """
    Result of a retrieval query with chunk and similarity score.
    """
    chunk: Chunk
    similarity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": self.similarity_score,
            "citation": self.chunk.to_citation(),
        }


def _extract_course_code_from_path(file_path: str) -> Optional[str]:
    """
    Extract course code from file path.
    
    Expected format: data/raw/<course_code>/...
    
    Args:
        file_path: Full path to the file
        
    Returns:
        Course code (e.g., "CS479") or None if not found
    """
    path = Path(file_path)
    parts = path.parts
    
    # Look for "raw" directory and get the next part (course code)
    if "raw" in parts:
        idx = parts.index("raw")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    
    # Fallback: try to find course code pattern (e.g., CS479, CS240)
    for part in parts:
        if part.startswith("CS") and len(part) >= 4:
            return part
    
    return None


def _chunk_model_to_chunk(chunk_model: ChunkModel) -> Chunk:
    """
    Convert ChunkModel database object to Chunk schema.
    
    Args:
        chunk_model: Database model instance
        
    Returns:
        Chunk object
    """
    # Parse locator from JSON
    locator_data = chunk_model.locator  # type: ignore
    locator_dict = locator_data if isinstance(locator_data, dict) else {}
    
    # Handle line_range - convert list to tuple if present
    line_range = None
    if "line_range" in locator_dict and locator_dict["line_range"] is not None:
        line_range_value = locator_dict["line_range"]
        if isinstance(line_range_value, (list, tuple)) and len(line_range_value) == 2:
            line_range = tuple(line_range_value)
    
    locator = ChunkLocator(
        page=locator_dict.get("page"),
        slide=locator_dict.get("slide"),
        section=locator_dict.get("section"),
        paragraph=locator_dict.get("paragraph"),
        line_range=line_range,
    )
    
    # Parse source type
    try:
        source_type = SourceType(chunk_model.source_type)  # type: ignore
    except ValueError:
        # Fallback to course_notes if invalid
        source_type = SourceType.COURSE_NOTES
    
    return Chunk(
        id=str(chunk_model.id),  # type: ignore
        file_path=str(chunk_model.file_path),  # type: ignore
        source_type=source_type,
        text=str(chunk_model.text),  # type: ignore
        locator=locator,
        chunk_index=int(chunk_model.chunk_index),  # type: ignore
        char_start=int(chunk_model.char_start) if chunk_model.char_start is not None else None,  # type: ignore
        char_end=int(chunk_model.char_end) if chunk_model.char_end is not None else None,  # type: ignore
        heading=str(chunk_model.heading) if chunk_model.heading is not None else None,  # type: ignore
    )


def retrieve_chunks(
    query: str,
    course_code: Optional[str] = None,
    source_types: Optional[List[SourceType]] = None,
    limit: int = 10,
    min_similarity: float = 0.0,
) -> List[RetrievalResult]:
    """
    Retrieve relevant chunks for a user query using vector similarity search.
    
    Args:
        query: User query text
        course_code: Optional course code to filter by (e.g., "CS479")
        source_types: Optional list of source types to filter by
        limit: Maximum number of results to return (default: 10)
        min_similarity: Minimum similarity score threshold (0.0 to 1.0, default: 0.0)
        
    Returns:
        List of RetrievalResult objects, sorted by similarity (highest first)
        
    Raises:
        ValueError: If embeddings cannot be generated (e.g., API key not set)
    """
    if not query or not query.strip():
        return []
    
    # Convert source types to strings if provided
    source_type_strings = None
    if source_types:
        source_type_strings = [st.value for st in source_types]
    
    # Build file path filter for course code if provided
    file_path_filter = None
    if course_code:
        # Filter by file path pattern: data/raw/<course_code>/%
        file_path_filter = f"%/raw/{course_code}/%"
    
    # Perform vector similarity search
    with VectorStore() as vector_store:
        # Query for similar chunks
        results = vector_store.query_similar(
            query_text=query,
            limit=limit * 2,  # Get more results to filter by min_similarity
            source_types=source_type_strings,
            min_similarity=min_similarity,
            file_path_filter=file_path_filter,
        )
    
    # Convert to Chunk objects and create RetrievalResult
    retrieval_results = []
    for chunk_model, similarity in results[:limit]:
        chunk = _chunk_model_to_chunk(chunk_model)
        retrieval_results.append(RetrievalResult(
            chunk=chunk,
            similarity_score=similarity,
        ))
    
    return retrieval_results


def retrieve_chunks_by_course(
    query: str,
    course_code: str,
    source_types: Optional[List[SourceType]] = None,
    limit: int = 10,
    min_similarity: float = 0.0,
) -> List[RetrievalResult]:
    """
    Convenience function to retrieve chunks for a specific course.
    
    Args:
        query: User query text
        course_code: Course code to filter by (e.g., "CS479")
        source_types: Optional list of source types to filter by
        limit: Maximum number of results to return (default: 10)
        min_similarity: Minimum similarity score threshold (0.0 to 1.0, default: 0.0)
        
    Returns:
        List of RetrievalResult objects, sorted by similarity (highest first)
    """
    return retrieve_chunks(
        query=query,
        course_code=course_code,
        source_types=source_types,
        limit=limit,
        min_similarity=min_similarity,
    )
