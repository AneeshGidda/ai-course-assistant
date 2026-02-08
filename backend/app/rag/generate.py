"""
RAG answer generation with citations.

Assembles retrieved chunks into context and generates answers using LLM,
ensuring all factual claims are properly cited.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .retrieve import retrieve_chunks, RetrievalResult, SourceType
from .prompts import (
    get_rag_system_prompt,
    format_rag_context,
    get_rag_user_prompt,
    get_teaching_prompt,
)
from app.services.llm import get_llm_service


@dataclass
class AnswerWithCitations:
    """
    Answer generated from RAG with citations.
    """
    answer: str
    citations: List[str]
    chunks_used: List[str]
    retrieval_results: List[RetrievalResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "chunks_used": self.chunks_used,
            "evidence": [
                {
                    "chunk_id": result.chunk.id,
                    "citation": result.chunk.to_citation(),
                    "similarity_score": result.similarity_score,
                    "text_preview": result.chunk.text[:200],
                }
                for result in self.retrieval_results
            ],
        }


def generate_answer(
    query: str,
    course_code: Optional[str] = None,
    source_types: Optional[List[SourceType]] = None,
    retrieval_limit: int = 10,
    min_similarity: float = 0.0,
    mode: str = "answer",  # "answer" or "teach"
) -> AnswerWithCitations:
    """
    Generate an answer using RAG with citations.
    
    This function:
    1. Retrieves relevant chunks using vector similarity
    2. Assembles chunks into RAG context
    3. Generates answer using LLM with citation requirements
    4. Extracts and validates citations
    
    Args:
        query: User's question
        course_code: Optional course code to filter by
        source_types: Optional list of source types to filter by
        retrieval_limit: Number of chunks to retrieve (default: 10)
        min_similarity: Minimum similarity threshold (default: 0.0)
        mode: Generation mode - "answer" (concise) or "teach" (explanatory)
        
    Returns:
        AnswerWithCitations object with answer, citations, and evidence
        
    Raises:
        ValueError: If LLM service is not available or no chunks retrieved
    """
    # Step 1: Retrieve relevant chunks
    retrieval_results = retrieve_chunks(
        query=query,
        course_code=course_code,
        source_types=source_types,
        limit=retrieval_limit,
        min_similarity=min_similarity,
    )
    
    if not retrieval_results:
        raise ValueError(
            f"No relevant chunks found for query: '{query}'. "
            "Try a different query or lower the min_similarity threshold."
        )
    
    # Step 2: Convert retrieval results to context format
    context_chunks = [result.to_dict() for result in retrieval_results]
    
    # Step 3: Format context for LLM
    context_text = format_rag_context(context_chunks)
    
    # Step 4: Get LLM service
    llm_service = get_llm_service()
    if not llm_service.is_available():
        raise ValueError(
            "LLM service is not available. Check OPENAI_API_KEY in your .env file."
        )
    
    # Step 5: Generate answer with appropriate prompt
    if mode == "teach":
        user_prompt = get_teaching_prompt(query, context_text)
        system_prompt = get_rag_system_prompt()
    else:
        user_prompt = get_rag_user_prompt(query, context_text)
        system_prompt = get_rag_system_prompt()
    
    # Generate answer
    llm_response = llm_service.generate_answer(
        query=query,
        context_chunks=context_chunks,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    
    # Step 6: Validate citations match retrieved chunks
    validated_citations = _validate_citations(
        llm_response["citations"],
        retrieval_results
    )
    
    # Step 7: Return answer with citations
    return AnswerWithCitations(
        answer=llm_response["answer"],
        citations=validated_citations,
        chunks_used=llm_response["chunks_used"],
        retrieval_results=retrieval_results,
    )


def _validate_citations(
    citations: List[str],
    retrieval_results: List[RetrievalResult]
) -> List[str]:
    """
    Validate that citations reference actual retrieved chunks.
    
    Args:
        citations: List of citation strings from LLM
        retrieval_results: List of retrieved chunks
        
    Returns:
        List of validated citations
    """
    # Build set of valid citations from retrieved chunks
    valid_citations = {result.chunk.to_citation() for result in retrieval_results}
    
    # Also check for partial matches (LLM might format slightly differently)
    validated = []
    for citation in citations:
        # Check exact match first
        if citation in valid_citations:
            validated.append(citation)
            continue
        
        # Check for partial match (case-insensitive)
        citation_lower = citation.lower()
        for valid_citation in valid_citations:
            if citation_lower in valid_citation.lower() or valid_citation.lower() in citation_lower:
                validated.append(valid_citation)
                break
    
    return validated


def generate_answer_with_evidence(
    query: str,
    course_code: Optional[str] = None,
    source_types: Optional[List[SourceType]] = None,
    retrieval_limit: int = 10,
    min_similarity: float = 0.0,
) -> Dict[str, Any]:
    """
    Convenience function that returns answer as dictionary with evidence.
    
    Args:
        query: User's question
        course_code: Optional course code to filter by
        source_types: Optional list of source types to filter by
        retrieval_limit: Number of chunks to retrieve
        min_similarity: Minimum similarity threshold
        
    Returns:
        Dictionary with answer, citations, and evidence chunks
    """
    answer = generate_answer(
        query=query,
        course_code=course_code,
        source_types=source_types,
        retrieval_limit=retrieval_limit,
        min_similarity=min_similarity,
    )
    
    return answer.to_dict()
