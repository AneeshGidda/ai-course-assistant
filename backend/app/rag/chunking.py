"""
Chunking strategies using LangChain text splitters.

Converts extracted document content into consistent Chunk representations
with appropriate locators (page, slide, section, etc.).
"""
import re
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from .schemas import Chunk, ChunkLocator, SourceType


def normalize_text(text: str) -> str:
    """
    Normalize chunk text for retrieval quality.
    
    - Remove NUL characters (PostgreSQL doesn't allow them)
    - Remove excessive whitespace
    - Normalize line breaks
    - Trim leading/trailing whitespace
    """
    # Remove NUL characters (0x00) - PostgreSQL doesn't allow them
    text = text.replace('\x00', '')
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def _create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a LangChain text splitter with appropriate settings.
    
    Args:
        chunk_size: Maximum size of chunks (in characters)
        chunk_overlap: Overlap between chunks (in characters)
        separators: List of separators to use (in order of preference)
    
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    if separators is None:
        # Default separators: try to split on paragraphs, sentences, then words
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )


def _langchain_docs_to_chunks(
    langchain_docs: List[LangChainDocument],
    file_path: str,
    source_type: SourceType,
    chunk_index_start: int = 0
) -> List[Chunk]:
    """
    Convert LangChain Document objects to our Chunk objects.
    
    Preserves metadata like page numbers, slide numbers, etc.
    """
    chunks = []
    chunk_idx = chunk_index_start
    
    for doc in langchain_docs:
        normalized = normalize_text(doc.page_content)
        if not normalized or len(normalized) < 10:
            continue
        
        # Extract metadata
        metadata = doc.metadata
        page = metadata.get("page")
        slide = metadata.get("slide")
        section = metadata.get("section")
        
        chunks.append(Chunk(
            file_path=file_path,
            source_type=source_type,
            text=normalized,
            locator=ChunkLocator(
                page=page,
                slide=slide,
                section=section,
            ),
            chunk_index=chunk_idx,
            heading=metadata.get("heading"),
        ))
        chunk_idx += 1
    
    return chunks


def chunk_by_paragraphs(
    text: str,
    file_path: str,
    source_type: SourceType,
    page: Optional[int] = None,
    section: Optional[str] = None,
) -> List[Chunk]:
    """
    Chunk text by paragraphs using LangChain RecursiveCharacterTextSplitter.
    
    For course_notes, student_notes, syllabus.
    """
    # Create splitter with paragraph-first strategy
    text_splitter = _create_text_splitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    # Create LangChain document with metadata
    doc = LangChainDocument(
        page_content=text,
        metadata={"page": page, "section": section}
    )
    
    # Split into chunks
    langchain_chunks = text_splitter.split_documents([doc])
    
    # Convert to our Chunk format
    return _langchain_docs_to_chunks(langchain_chunks, file_path, source_type)


def chunk_by_slides(
    slides: List[str],
    file_path: str,
    source_type: SourceType,
    page: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk by slides (for lecture_slides).
    
    Each slide becomes a chunk with slide number as locator.
    """
    chunks = []
    
    for idx, slide_text in enumerate(slides):
        normalized = normalize_text(slide_text)
        if not normalized or len(normalized) < 10:
            continue
        
        chunks.append(Chunk(
            file_path=file_path,
            source_type=source_type,
            text=normalized,
            locator=ChunkLocator(
                page=page,
                slide=idx + 1,
            ),
            chunk_index=len(chunks),
        ))
    
    return chunks


def chunk_by_sections(
    sections: dict[str, str],
    file_path: str,
    source_type: SourceType,
    page: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk by sections (for course_notes with clear structure).
    
    Each section is chunked using LangChain, maintaining section context.
    """
    all_chunks = []
    chunk_idx = 0
    
    # Use paragraph-based splitter for sections
    text_splitter = _create_text_splitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    for section_name, section_text in sections.items():
        # Create document with section metadata
        doc = LangChainDocument(
            page_content=section_text,
            metadata={"page": page, "section": section_name, "heading": section_name}
        )
        
        # Split section into chunks
        section_chunks = text_splitter.split_documents([doc])
        
        # Convert to our Chunk format
        for lc_chunk in section_chunks:
            normalized = normalize_text(lc_chunk.page_content)
            if not normalized or len(normalized) < 10:
                continue
            
            all_chunks.append(Chunk(
                file_path=file_path,
                source_type=source_type,
                text=normalized,
                locator=ChunkLocator(
                    page=page,
                    section=section_name,
                ),
                chunk_index=chunk_idx,
                heading=section_name,
            ))
            chunk_idx += 1
    
    return all_chunks


def chunk_problem_set(
    text: str,
    file_path: str,
    source_type: SourceType,
    page: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk problem sets (for practice_problems, exams, assignments).
    
    First tries to identify problems, then chunks each problem separately.
    Falls back to paragraph chunking if no problems found.
    """
    # Try to identify problems by common patterns
    problem_pattern = re.compile(
        r'(?:Problem|Question|Exercise|Part)\s*(\d+)[:.]?\s*\n',
        re.IGNORECASE
    )
    
    # Split by problem markers
    parts = problem_pattern.split(text)
    
    # First part might be preamble (skip if short)
    current_problem_num = None
    current_text = []
    problems = []
    
    for i, part in enumerate(parts):
        # Check if this part is a problem number
        if i % 2 == 1 and part.isdigit():
            # Save previous problem if exists
            if current_problem_num is not None and current_text:
                problems.append({
                    "number": current_problem_num,
                    "text": '\n'.join(current_text)
                })
            
            # Start new problem
            current_problem_num = int(part)
            current_text = []
        else:
            current_text.append(part.strip())
    
    # Add last problem
    if current_problem_num is not None and current_text:
        problems.append({
            "number": current_problem_num,
            "text": '\n'.join(current_text)
        })
    
    # If we found problems, chunk each one separately
    if problems:
        all_chunks = []
        chunk_idx = 0
        
        # Use smaller chunk size for problems (they're usually self-contained)
        text_splitter = _create_text_splitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        for problem in problems:
            problem_text = problem["text"]
            problem_num = problem["number"]
            
            # Create document with problem metadata
            doc = LangChainDocument(
                page_content=problem_text,
                metadata={
                    "page": page,
                    "section": f"Problem {problem_num}",
                    "heading": f"Problem {problem_num}"
                }
            )
            
            # Split problem into chunks (if it's long enough)
            problem_chunks = text_splitter.split_documents([doc])
            
            for lc_chunk in problem_chunks:
                normalized = normalize_text(lc_chunk.page_content)
                if not normalized or len(normalized) < 10:
                    continue
                
                all_chunks.append(Chunk(
                    file_path=file_path,
                    source_type=source_type,
                    text=normalized,
                    locator=ChunkLocator(
                        page=page,
                        section=f"Problem {problem_num}",
                    ),
                    chunk_index=chunk_idx,
                    heading=f"Problem {problem_num}",
                ))
                chunk_idx += 1
        
        return all_chunks
    
    # Fallback: chunk by paragraphs if no problems found
    return chunk_by_paragraphs(text, file_path, source_type, page=page)


def chunk_by_source_type(
    text: str,
    file_path: str,
    source_type: SourceType,
    page: Optional[int] = None,
    slides: Optional[List[str]] = None,
    sections: Optional[dict[str, str]] = None,
) -> List[Chunk]:
    """
    Apply appropriate chunking strategy based on source_type using LangChain.
    
    Args:
        text: Extracted document text
        file_path: Path to source file
        source_type: Type of source document
        page: Page number if applicable
        slides: List of slide texts (for PPTX)
        sections: Dictionary of section_name -> section_text
        
    Returns:
        List of Chunk objects
    """
    if source_type == SourceType.LECTURE_SLIDES:
        if slides:
            return chunk_by_slides(slides, file_path, source_type, page=page)
        # Fallback: treat as paragraphs
        return chunk_by_paragraphs(text, file_path, source_type, page=page)
    
    elif source_type in (SourceType.PRACTICE_PROBLEMS, SourceType.EXAM, SourceType.ASSIGNMENT):
        return chunk_problem_set(text, file_path, source_type, page=page)
    
    elif source_type == SourceType.COURSE_NOTES:
        # Try sections first if available
        if sections:
            return chunk_by_sections(sections, file_path, source_type, page=page)
        # Fallback to paragraphs
        return chunk_by_paragraphs(text, file_path, source_type, page=page)
    
    elif source_type in (SourceType.STUDENT_NOTES, SourceType.SYLLABUS, SourceType.SOLUTION):
        return chunk_by_paragraphs(text, file_path, source_type, page=page)
    
    else:
        # Default: chunk by paragraphs
        return chunk_by_paragraphs(text, file_path, source_type, page=page)
