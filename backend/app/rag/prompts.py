"""
Professor prompts for RAG answer generation.

Defines system prompts and templates for generating answers with citations.
"""
from typing import List, Dict, Any


def get_rag_system_prompt() -> str:
    """
    Get the system prompt for RAG answer generation.
    
    Enforces citation requirements and factual grounding.
    """
    return """You are a helpful professor assistant answering questions about course materials.

CRITICAL RULES:
1. ONLY use information from the provided course materials - do not use external knowledge
2. For EVERY factual claim, include a citation in the format: [Citation: filename, location]
3. If the materials don't contain enough information to answer, say "Based on the provided materials, I cannot fully answer this question."
4. Be clear, educational, and precise in your explanations
5. When citing, use the exact format: [Citation: filename, location] where location is the page/slide/section number

Example citations:
- [Citation: lecture01.pdf, page 5]
- [Citation: notes.pdf, slide 3]
- [Citation: main.pdf, Section 2.1]

Remember: Every factual statement must have a citation. If you're not certain, say so."""


def format_rag_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into RAG context.
    
    Args:
        chunks: List of RetrievalResult dictionaries with chunk and citation info
        
    Returns:
        Formatted context string for the LLM
    """
    context_parts = []
    
    for i, chunk_data in enumerate(chunks, 1):
        chunk = chunk_data.get("chunk", {})
        citation = chunk_data.get("citation", f"Source {i}")
        similarity = chunk_data.get("similarity_score", 0.0)
        text = chunk.get("text", "")
        
        # Include similarity score for reference (optional)
        context_parts.append(
            f"[Source {i}] {citation} (relevance: {similarity:.2f})\n"
            f"{text}\n"
        )
    
    return "\n".join(context_parts)


def get_rag_user_prompt(query: str, context: str) -> str:
    """
    Get the user prompt for RAG answer generation.
    
    Args:
        query: User's question
        context: Formatted context from retrieved chunks
        
    Returns:
        User prompt string
    """
    return f"""Based on the following course materials, answer the user's question.

Course Materials:
{context}

User Question: {query}

Remember:
- Only use information from the materials above
- Cite every factual claim using [Citation: filename, location] format
- If you're uncertain, say so
- Be educational and clear"""


def get_teaching_prompt(query: str, context: str) -> str:
    """
    Get prompt for teaching mode (more explanatory).
    
    Args:
        query: User's question
        context: Formatted context from retrieved chunks
        
    Returns:
        Teaching prompt string
    """
    return f"""You are teaching a student about this topic. Based on the course materials below, provide a clear, step-by-step explanation.

Course Materials:
{context}

Student's Question: {query}

Provide a teaching-style explanation that:
1. Builds understanding step by step
2. Uses examples from the materials
3. Cites sources for all facts: [Citation: filename, location]
4. Connects concepts together
5. Checks for understanding"""


def get_exam_question_prompt(topic: str, context: str) -> str:
    """
    Get prompt for generating exam questions.
    
    Args:
        topic: Topic to generate question about
        context: Formatted context from retrieved chunks
        
    Returns:
        Exam question generation prompt
    """
    return f"""Based on the course materials below, generate a practice exam question about: {topic}

Course Materials:
{context}

Generate a question that:
1. Tests understanding of key concepts
2. Is appropriate for the course level
3. Includes a detailed solution with citations: [Citation: filename, location]
4. References specific material from the course"""
