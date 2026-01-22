"""
Structured LLM outputs and data schemas for RAG ingestion.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4


class SourceType(str, Enum):
    """Semantic source type enum. This is NOT inferred from content - it's explicitly set."""
    COURSE_NOTES = "course_notes"
    LECTURE_SLIDES = "lecture_slides"
    STUDENT_NOTES = "student_notes"
    SYLLABUS = "syllabus"
    PRACTICE_PROBLEMS = "practice_problems"
    EXAM = "exam"
    SOLUTION = "solution"
    ASSIGNMENT = "assignment"


@dataclass
class ChunkLocator:
    """
    Stable locator for referencing a chunk's position in a document.
    Provides deterministic citation information.
    """
    # Page number (1-indexed), if applicable
    page: Optional[int] = None
    
    # Slide number (1-indexed), if applicable
    slide: Optional[int] = None
    
    # Section number/name, if applicable
    section: Optional[str] = None
    
    # Paragraph number (1-indexed), if applicable
    paragraph: Optional[int] = None
    
    # Line number range, if applicable (tuple of start, end)
    line_range: Optional[tuple[int, int]] = None
    
    def to_dict(self) -> dict:
        """Convert locator to dictionary for serialization."""
        result = {}
        if self.page is not None:
            result["page"] = self.page
        if self.slide is not None:
            result["slide"] = self.slide
        if self.section is not None:
            result["section"] = self.section
        if self.paragraph is not None:
            result["paragraph"] = self.paragraph
        if self.line_range is not None:
            result["line_range"] = list(self.line_range)
        return result
    
    def to_citation(self) -> str:
        """Generate human-readable citation string."""
        parts = []
        if self.section:
            parts.append(f"Section {self.section}")
        if self.page:
            parts.append(f"page {self.page}")
        if self.slide:
            parts.append(f"slide {self.slide}")
        if self.paragraph:
            parts.append(f"paragraph {self.paragraph}")
        if self.line_range:
            parts.append(f"lines {self.line_range[0]}-{self.line_range[1]}")
        
        if not parts:
            return "unknown location"
        
        return ", ".join(parts)


@dataclass
class Chunk:
    """
    Unified chunk representation for retrieval and citation.
    
    All chunks follow this consistent schema regardless of source type.
    """
    # Source file information
    file_path: str
    source_type: SourceType
    
    # Chunk content (normalized text)
    text: str
    
    # Locator information for citation
    locator: ChunkLocator
    
    # Chunk metadata
    chunk_index: int  # Order within document (0-indexed)
    
    # Unique identifier for this chunk
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Optional: start/end character positions in original document
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    
    # Optional: heading/section title that this chunk belongs to
    heading: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "source_type": self.source_type.value,
            "text": self.text,
            "locator": self.locator.to_dict(),
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "heading": self.heading,
        }
    
    def to_citation(self) -> str:
        """Generate human-readable citation for this chunk."""
        file_name = Path(self.file_path).name
        location = self.locator.to_citation()
        return f"{file_name}, {location}"


# Allowed file format constants
ALLOWED_FORMATS = {
    ".pdf": "slides, notes, exams, syllabus",
    ".pptx": "lecture slides (best structured)",
    ".docx": "notes, worksheets",
    ".md": "clean notes",
    ".txt": "clean notes",
}

# Format → source_type mappings (strict validation)
FORMAT_TO_SOURCE_TYPE: dict[str, set[SourceType]] = {
    ".pdf": {
        SourceType.COURSE_NOTES,
        SourceType.LECTURE_SLIDES,
        SourceType.STUDENT_NOTES,
        SourceType.SYLLABUS,
        SourceType.PRACTICE_PROBLEMS,
        SourceType.EXAM,
        SourceType.SOLUTION,
        SourceType.ASSIGNMENT,
    },
    ".pptx": {
        SourceType.LECTURE_SLIDES,
    },
    ".docx": {
        SourceType.COURSE_NOTES,
        SourceType.STUDENT_NOTES,
        SourceType.SYLLABUS,
        SourceType.PRACTICE_PROBLEMS,
        SourceType.SOLUTION,
        SourceType.ASSIGNMENT,
    },
    ".md": {
        SourceType.COURSE_NOTES,
        SourceType.STUDENT_NOTES,
    },
    ".txt": {
        SourceType.COURSE_NOTES,
        SourceType.STUDENT_NOTES,
    },
}
