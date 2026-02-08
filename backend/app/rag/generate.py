"""
RAG answer generation with citations.

Assembles retrieved chunks into context and generates answers using LLM,
ensuring all factual claims are properly cited.
"""
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .retrieve import retrieve_chunks, RetrievalResult, SourceType
from .prompts import (
    get_rag_system_prompt,
    format_rag_context,
    get_rag_user_prompt,
    get_teaching_prompt,
)
from .exceptions import InsufficientMaterialError
from app.services.llm import get_llm_service
from app.config import MIN_RETRIEVAL_CONFIDENCE, MIN_HIGH_QUALITY_CHUNKS


@dataclass
class AnswerWithCitations:
    """
    Answer generated from RAG with citations.
    """
    answer: str
    citations: List[str]
    chunks_used: List[str]
    retrieval_results: List[RetrievalResult]
    retrieval_quality: Optional[Dict[str, Any]] = None  # Quality metrics
    
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
            "retrieval_quality": self.retrieval_quality,
        }


def _assess_retrieval_quality(
    retrieval_results: List[RetrievalResult],
    min_confidence: float = MIN_RETRIEVAL_CONFIDENCE,
    min_high_quality_chunks: int = MIN_HIGH_QUALITY_CHUNKS,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Assess the quality of retrieved chunks to prevent hallucinations.
    
    Args:
        retrieval_results: List of retrieved chunks with similarity scores
        min_confidence: Minimum similarity score for top result (default: from config)
        min_high_quality_chunks: Minimum number of chunks above threshold (default: from config)
        
    Returns:
        Tuple of (is_sufficient, quality_metrics)
        - is_sufficient: True if quality is high enough to generate answer
        - quality_metrics: Dictionary with quality statistics
    """
    if not retrieval_results:
        return False, {
            "top_similarity": 0.0,
            "high_quality_chunks": 0,
            "total_chunks": 0,
            "is_sufficient": False,
        }
    
    # Get top similarity score
    top_similarity = retrieval_results[0].similarity_score if retrieval_results else 0.0
    
    # Count chunks above confidence threshold
    high_quality_chunks = sum(
        1 for result in retrieval_results
        if result.similarity_score >= min_confidence
    )
    
    quality_metrics = {
        "top_similarity": top_similarity,
        "high_quality_chunks": high_quality_chunks,
        "total_chunks": len(retrieval_results),
        "min_confidence_threshold": min_confidence,
        "min_high_quality_chunks_required": min_high_quality_chunks,
    }
    
    # Check if quality is sufficient
    # Need: (1) top result above threshold AND (2) enough high-quality chunks
    is_sufficient = (
        top_similarity >= min_confidence and
        high_quality_chunks >= min_high_quality_chunks
    )
    
    quality_metrics["is_sufficient"] = is_sufficient
    
    return is_sufficient, quality_metrics


def generate_answer(
    query: str,
    course_code: Optional[str] = None,
    source_types: Optional[List[SourceType]] = None,
    retrieval_limit: int = 10,
    min_similarity: float = 0.0,
    mode: str = "answer",  # "answer" or "teach"
    min_confidence: Optional[float] = None,  # Override default from config
    min_high_quality_chunks: Optional[int] = None,  # Override default from config
    strict_quality_check: bool = True,  # If False, allows lower quality answers
) -> AnswerWithCitations:
    """
    Generate an answer using RAG with citations.
    
    This function:
    1. Retrieves relevant chunks using vector similarity
    2. Assesses retrieval quality to prevent hallucinations
    3. Assembles chunks into RAG context
    4. Generates answer using LLM with citation requirements
    5. Extracts and validates citations
    
    Args:
        query: User's question
        course_code: Optional course code to filter by
        source_types: Optional list of source types to filter by
        retrieval_limit: Number of chunks to retrieve (default: 10)
        min_similarity: Minimum similarity threshold for retrieval (default: 0.0)
        mode: Generation mode - "answer" (concise) or "teach" (explanatory)
        min_confidence: Minimum similarity for top result (default: from config)
        min_high_quality_chunks: Minimum high-quality chunks required (default: from config)
        strict_quality_check: If True, refuses answers when quality is low (default: True)
        
    Returns:
        AnswerWithCitations object with answer, citations, and evidence
        
    Raises:
        InsufficientMaterialError: If retrieval quality is too low (when strict_quality_check=True)
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
    
    # Step 2: Assess retrieval quality to prevent hallucinations
    confidence_threshold = min_confidence if min_confidence is not None else MIN_RETRIEVAL_CONFIDENCE
    high_quality_threshold = min_high_quality_chunks if min_high_quality_chunks is not None else MIN_HIGH_QUALITY_CHUNKS
    
    is_sufficient, quality_metrics = _assess_retrieval_quality(
        retrieval_results,
        min_confidence=confidence_threshold,
        min_high_quality_chunks=high_quality_threshold,
    )
    
    # Step 3: Refuse answer if quality is insufficient (unless strict check is disabled)
    if strict_quality_check and not is_sufficient:
        # Generate clear refusal message
        top_sim = quality_metrics["top_similarity"]
        high_quality = quality_metrics["high_quality_chunks"]
        total = quality_metrics["total_chunks"]
        
        message = (
            f"I don't have sufficient information in the course materials to answer this question reliably. "
            f"The retrieved content has a similarity score of {top_sim:.2f} (minimum required: {confidence_threshold:.2f}), "
            f"and only {high_quality} out of {total} retrieved chunks meet the quality threshold "
            f"(minimum required: {high_quality_threshold}). "
            f"To prevent providing inaccurate information, I cannot answer this question based on the available materials. "
            f"Please try rephrasing your question or asking about a different topic."
        )
        
        raise InsufficientMaterialError(
            query=query,
            top_similarity=top_sim,
            high_quality_chunks=high_quality,
            total_chunks=total,
            message=message,
        )
    
    # Step 4: Convert retrieval results to context format
    context_chunks = [result.to_dict() for result in retrieval_results]
    
    # Step 5: Format context for LLM
    context_text = format_rag_context(context_chunks)
    
    # Step 6: Get LLM service
    llm_service = get_llm_service()
    if not llm_service.is_available():
        raise ValueError(
            "LLM service is not available. Check OPENAI_API_KEY in your .env file."
        )
    
    # Step 7: Generate answer with appropriate prompt
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
    
    # Step 8: Validate citations match retrieved chunks
    validated_citations = _validate_citations(
        llm_response["citations"],
        retrieval_results
    )
    
    # Step 9: Return answer with citations and quality metrics
    return AnswerWithCitations(
        answer=llm_response["answer"],
        citations=validated_citations,
        chunks_used=llm_response["chunks_used"],
        retrieval_results=retrieval_results,
        retrieval_quality=quality_metrics,
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
    strict_quality_check: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function that returns answer as dictionary with evidence.
    
    Args:
        query: User's question
        course_code: Optional course code to filter by
        source_types: Optional list of source types to filter by
        retrieval_limit: Number of chunks to retrieve
        min_similarity: Minimum similarity threshold
        strict_quality_check: If True, refuses answers when quality is low
        
    Returns:
        Dictionary with answer, citations, and evidence chunks
        
    Raises:
        InsufficientMaterialError: If retrieval quality is too low (when strict_quality_check=True)
    """
    answer = generate_answer(
        query=query,
        course_code=course_code,
        source_types=source_types,
        retrieval_limit=retrieval_limit,
        min_similarity=min_similarity,
        strict_quality_check=strict_quality_check,
    )
    
    return answer.to_dict()
