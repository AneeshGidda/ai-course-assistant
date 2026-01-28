"""
Embedding client for generating vector embeddings.
"""
from typing import List, Optional
import os

from langchain_openai import OpenAIEmbeddings

from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI.
    
    Handles API key validation and provides a simple interface for embedding text.
    """
    
    def __init__(self):
        """Initialize the embedding service."""
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._initialized = False
    
    def _initialize(self) -> bool:
        """
        Initialize the OpenAI embeddings client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return self._embeddings is not None
        
        self._initialized = True
        
        # Check for API key
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set. Embeddings will not be generated.")
            return False
        
        try:
            # Set environment variable for OpenAI client
            os.environ["OPENAI_API_KEY"] = api_key
            self._embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
            )
            return True
        except Exception as e:
            print(f"WARNING: Failed to initialize embeddings: {e}")
            return False
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if embedding failed
        """
        if not self._initialize():
            return None
        
        if not self._embeddings:
            return None
        
        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            print(f"WARNING: Failed to generate embedding: {e}")
            return None
    
    def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (or None for failed embeddings)
        """
        if not self._initialize():
            return [None] * len(texts)
        
        if not self._embeddings:
            return [None] * len(texts)
        
        try:
            # Use embed_documents for batch processing (more efficient)
            embeddings = self._embeddings.embed_documents(texts)
            # Convert to List[Optional[List[float]]] for type consistency
            return [emb for emb in embeddings]  # type: ignore
        except Exception as e:
            print(f"WARNING: Failed to generate embeddings: {e}")
            return [None] * len(texts)
    
    def is_available(self) -> bool:
        """
        Check if embedding service is available.
        
        Returns:
            True if embeddings can be generated, False otherwise
        """
        return self._initialize()


# Global singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
