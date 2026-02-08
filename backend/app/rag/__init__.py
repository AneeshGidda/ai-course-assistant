"""RAG (Retrieval Augmented Generation) module for course material ingestion and retrieval."""

from .chunking import (
    chunk_by_source_type,
    chunk_by_paragraphs,
    chunk_by_slides,
    chunk_by_sections,
    chunk_problem_set,
    normalize_text,
)
from .schemas import (
    SourceType,
    Chunk,
    ChunkLocator,
    ALLOWED_FORMATS,
    FORMAT_TO_SOURCE_TYPE,
)
from .validation import (
    IngestionValidationError,
    validate_format,
    validate_format_source_type_mapping,
    infer_source_type_from_path,
    validate_file_for_ingestion,
)
from .ingest import ingest_file, ingest_course, discover_course_files
from .parsing import parse_file, parse_pdf, parse_pptx, parse_docx, parse_text_file
from .vector_store import VectorStore
from .retrieve import (
    retrieve_chunks,
    retrieve_chunks_by_course,
    RetrievalResult,
)
from .generate import (
    generate_answer,
    generate_answer_with_evidence,
    AnswerWithCitations,
)
from .exceptions import InsufficientMaterialError

__all__ = [
    "SourceType",
    "Chunk",
    "ChunkLocator",
    "ALLOWED_FORMATS",
    "FORMAT_TO_SOURCE_TYPE",
    "IngestionValidationError",
    "validate_format",
    "validate_format_source_type_mapping",
    "infer_source_type_from_path",
    "validate_file_for_ingestion",
    "ingest_file",
    "ingest_course",
    "discover_course_files",
    "chunk_by_source_type",
    "chunk_by_paragraphs",
    "chunk_by_slides",
    "chunk_by_sections",
    "chunk_problem_set",
    "normalize_text",
    "parse_file",
    "parse_pdf",
    "parse_pptx",
    "parse_docx",
    "parse_text_file",
    "VectorStore",
    "retrieve_chunks",
    "retrieve_chunks_by_course",
    "RetrievalResult",
    "generate_answer",
    "generate_answer_with_evidence",
    "AnswerWithCitations",
    "InsufficientMaterialError",
]
