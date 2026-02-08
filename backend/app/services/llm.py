"""
LLM client wrapper for generating answers with citations.
"""
from typing import List, Optional, Dict, Any
import os

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


class LLMService:
    """
    Service for generating answers using OpenAI's chat models.
    
    Handles API key validation and provides a simple interface for chat completion.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self._client: Optional[OpenAI] = None
        self._chat_model: Optional[ChatOpenAI] = None
        self._initialized = False
    
    def _initialize(self) -> bool:
        """
        Initialize the OpenAI client and chat model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return self._client is not None
        
        self._initialized = True
        
        # Check for API key
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set. LLM will not be available.")
            return False
        
        try:
            # Set environment variable for OpenAI client
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Initialize OpenAI client
            self._client = OpenAI(api_key=api_key)
            
            # Initialize LangChain chat model
            self._chat_model = ChatOpenAI(
                model="gpt-4o-mini",  # Cost-effective model, can switch to gpt-4o for better quality
                temperature=0.3,  # Lower temperature for more factual responses
            )
            
            return True
        except Exception as e:
            print(f"WARNING: Failed to initialize LLM: {e}")
            return False
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved chunks with citations.
        
        Args:
            query: User's question
            context_chunks: List of chunk dictionaries with text and metadata
            system_prompt: Optional custom system prompt
            user_prompt: Optional custom user prompt (if provided, query and context_chunks are ignored)
            
        Returns:
            Dictionary with:
                - answer: Generated answer text
                - citations: List of citations used
                - chunks_used: List of chunk IDs used in the answer
                
        Raises:
            ValueError: If LLM is not available
        """
        if not self._initialize():
            raise ValueError("LLM service is not available. Check OPENAI_API_KEY.")
        
        if not self._chat_model:
            raise ValueError("Chat model not initialized.")
        
        # Build messages
        messages = []
        
        # System prompt
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        else:
            messages.append(SystemMessage(content=self._get_default_system_prompt()))
        
        # User message
        if user_prompt:
            messages.append(HumanMessage(content=user_prompt))
        else:
            # Build context from chunks
            context_text = self._build_context(context_chunks)
            
            # User query with context
            user_message = f"""Based on the following course materials, answer the user's question.

Course Materials:
{context_text}

User Question: {query}

Remember to cite your sources using [Citation: filename, location] format for all factual claims."""
            
            messages.append(HumanMessage(content=user_message))
        
        # Generate response
        try:
            response = self._chat_model.invoke(messages)
            answer_text = response.content
            
            # Extract citations from the answer
            citations = self._extract_citations(answer_text, context_chunks)
            
            # Get chunk IDs that were used (based on citations)
            chunks_used = [c["chunk"]["id"] for c in context_chunks if c.get("citation") in citations]
            
            return {
                "answer": answer_text,
                "citations": citations,
                "chunks_used": chunks_used,
            }
        except Exception as e:
            raise ValueError(f"Failed to generate answer: {e}")
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk_data in enumerate(chunks, 1):
            chunk = chunk_data.get("chunk", {})
            citation = chunk_data.get("citation", f"Source {i}")
            text = chunk.get("text", "")
            
            context_parts.append(f"[{i}] {citation}\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt that enforces citations."""
        return """You are a helpful professor assistant answering questions about course materials.

IMPORTANT RULES:
1. Only use information from the provided course materials
2. For EVERY factual claim, include a citation in the format: [Citation: filename, location]
3. If you don't know the answer based on the materials, say so
4. Be clear and educational in your explanations
5. Use the exact citation format shown in the materials

Example citation format: [Citation: lecture01.pdf, page 5] or [Citation: notes.pdf, slide 3]"""
    
    def _extract_citations(self, answer_text: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Extract citations from the answer text.
        
        Args:
            answer_text: Generated answer text
            chunks: List of chunk dictionaries for reference
            
        Returns:
            List of unique citations found in the answer
        """
        import re
        
        # Pattern to match [Citation: ...] format
        citation_pattern = r'\[Citation:\s*([^\]]+)\]'
        citations = re.findall(citation_pattern, answer_text, re.IGNORECASE)
        
        # Return unique citations
        return list(set(citations))
    
    def is_available(self) -> bool:
        """
        Check if LLM service is available.
        
        Returns:
            True if LLM can be used, False otherwise
        """
        return self._initialize()


# Global singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get the global LLM service instance.
    
    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
