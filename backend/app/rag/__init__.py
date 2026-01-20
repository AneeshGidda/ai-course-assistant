"""RAG (Retrieval Augmented Generation) module for course material ingestion and retrieval."""

from .schemas import SourceType, ALLOWED_FORMATS, FORMAT_TO_SOURCE_TYPE
from .validation import (
    IngestionValidationError,
    validate_format,
    validate_format_source_type_mapping,
    infer_source_type_from_path,
    validate_file_for_ingestion,
)
from .ingest import ingest_file, ingest_course, discover_course_files

__all__ = [
    "SourceType",
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
]
