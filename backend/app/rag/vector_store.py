"""
Vector store for persisting and querying chunks with embeddings.
"""
from typing import List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except ImportError:
    # Fallback for type checking
    Vector = None

from app.db.database import SessionLocal
from app.models.course import ChunkModel
from app.rag.schemas import Chunk
from app.services.embeddings import get_embedding_service


class VectorStore:
    """
    Vector store for storing and querying chunks with embeddings.
    
    Uses pgvector for efficient similarity search.
    Supports deduplication based on file_path and chunk content.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize vector store.
        
        Args:
            db_session: Optional database session (creates new if not provided)
        """
        self.db = db_session or SessionLocal()
        self.embedding_service = get_embedding_service()
    
    def store_chunks(
        self,
        chunks: List[Chunk],
        generate_embeddings: bool = True
    ) -> Tuple[int, int]:
        """
        Store chunks in the vector database.
        
        Args:
            chunks: List of Chunk objects to store
            generate_embeddings: Whether to generate embeddings (default: True)
            
        Returns:
            Tuple of (stored_count, skipped_count)
        """
        if not chunks:
            return 0, 0
        
        stored_count = 0
        skipped_count = 0
        
        # Generate embeddings if requested and available
        embeddings: List[Optional[List[float]]] = []
        if generate_embeddings and self.embedding_service.is_available():
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Debug: Count how many embeddings were generated
            successful_embeddings = sum(1 for emb in embeddings if emb is not None)
            if successful_embeddings == 0:
                print(f"WARNING: No embeddings were generated for {len(chunks)} chunks")
            elif successful_embeddings < len(chunks):
                print(f"WARNING: Only {successful_embeddings}/{len(chunks)} embeddings were generated")
        else:
            embeddings = [None] * len(chunks)
            if generate_embeddings:
                print("WARNING: Embedding service is not available. Chunks will be stored without embeddings.")
        
        now = datetime.utcnow().isoformat()
        
        for chunk, embedding in zip(chunks, embeddings):
            # Check if chunk already exists (deduplication)
            existing = self._find_existing_chunk(chunk)
            
            if existing:
                # Clean text to remove NUL characters before storing
                clean_text = chunk.text.replace('\x00', '') if chunk.text else ''
                
                # Update existing chunk if needed
                if embedding is not None:
                    existing.embedding = embedding  # type: ignore
                existing.updated_at = now  # type: ignore
                existing.text = clean_text  # type: ignore
                existing.locator = chunk.locator.to_dict()  # type: ignore
                existing.chunk_index = chunk.chunk_index  # type: ignore
                existing.char_start = chunk.char_start  # type: ignore
                existing.char_end = chunk.char_end  # type: ignore
                existing.heading = chunk.heading  # type: ignore
                skipped_count += 1
            else:
                # Clean text to remove NUL characters before storing
                clean_text = chunk.text.replace('\x00', '') if chunk.text else ''
                
                # Create new chunk
                chunk_model = ChunkModel(
                    id=chunk.id,
                    file_path=chunk.file_path,
                    source_type=chunk.source_type.value,
                    text=clean_text,
                    locator=chunk.locator.to_dict(),
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    heading=chunk.heading,
                    embedding=embedding,
                    created_at=now,
                    updated_at=now,
                )
                self.db.add(chunk_model)
                stored_count += 1
        
        try:
            self.db.commit()
            
            # Debug: Verify embeddings were stored
            if generate_embeddings:
                with_embeddings = self.db.query(ChunkModel).filter(
                    ChunkModel.embedding.isnot(None)
                ).count()
                if with_embeddings == 0 and stored_count > 0:
                    print(f"WARNING: {stored_count} chunks stored but none have embeddings.")
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to store chunks: {e}")
        
        return stored_count, skipped_count
    
    def _find_existing_chunk(self, chunk: Chunk) -> Optional[ChunkModel]:
        """
        Find existing chunk by file_path and chunk content hash.
        
        Uses a combination of file_path and normalized text for deduplication.
        
        Args:
            chunk: Chunk to search for
            
        Returns:
            Existing ChunkModel if found, None otherwise
        """
        # First, try to find by exact file_path and chunk_index
        existing = self.db.query(ChunkModel).filter(
            ChunkModel.file_path == chunk.file_path,
            ChunkModel.chunk_index == chunk.chunk_index,
        ).first()
        
        if existing:
            return existing
        
        # Fallback: find by file_path and text similarity (normalized)
        # This handles cases where chunk_index might have changed
        normalized_text = chunk.text.strip().replace('\n', ' ').replace('  ', ' ')
        existing = self.db.query(ChunkModel).filter(
            ChunkModel.file_path == chunk.file_path,
            ChunkModel.text.ilike(f"%{normalized_text[:100]}%")  # Partial match on first 100 chars
        ).first()
        
        return existing
    
    def query_similar(
        self,
        query_text: str,
        limit: int = 10,
        source_types: Optional[List[str]] = None,
        min_similarity: float = 0.0
    ) -> List[Tuple[ChunkModel, float]]:
        """
        Query for similar chunks using vector similarity search.
        
        Args:
            query_text: Query text to search for
            limit: Maximum number of results
            source_types: Optional list of source types to filter by
            min_similarity: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of tuples (ChunkModel, similarity_score)
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query_text)
        if not query_embedding:
            return []
        
        # Build query
        query = self.db.query(ChunkModel).filter(
            ChunkModel.embedding.isnot(None)
        )
        
        # Filter by source types if provided
        if source_types:
            query = query.filter(ChunkModel.source_type.in_(source_types))
        
        # Vector similarity search using cosine distance
        # Order by cosine distance (ascending = most similar first)
        results = query.order_by(
            ChunkModel.embedding.cosine_distance(query_embedding)
        ).limit(limit * 2).all()  # Get more results to filter by min_similarity
        
        # Calculate similarity scores and filter
        results_with_scores = []
        for result in results:
            if result.embedding is not None:  # type: ignore
                # Calculate cosine similarity (1 - distance)
                distance = float(result.embedding.cosine_distance(query_embedding))  # type: ignore
                similarity = 1 - distance
                if similarity >= min_similarity:
                    results_with_scores.append((result, similarity))
        
        # Sort by similarity (descending) - already sorted by distance, but ensure order
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return results_with_scores[:limit]
    
    def delete_chunks_by_file(self, file_path: str) -> int:
        """
        Delete all chunks for a specific file.
        
        Useful for re-ingestion: delete old chunks before storing new ones.
        
        Args:
            file_path: Path to file whose chunks should be deleted
            
        Returns:
            Number of chunks deleted
        """
        count = self.db.query(ChunkModel).filter(
            ChunkModel.file_path == file_path
        ).delete()
        
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Failed to delete chunks: {e}")
        
        return count
    
    def get_chunk_count(self, file_path: Optional[str] = None) -> int:
        """
        Get total number of chunks in the database.
        
        Args:
            file_path: Optional file path to filter by
            
        Returns:
            Number of chunks
        """
        query = self.db.query(ChunkModel)
        if file_path:
            query = query.filter(ChunkModel.file_path == file_path)
        return query.count()
    
    def close(self):
        """Close the database session."""
        self.db.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
