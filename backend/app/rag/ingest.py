"""
PDFs → chunks → embeddings

Ingestion pipeline with strict format and source_type validation.
"""
from pathlib import Path
from typing import List, Optional

from .validation import validate_file_for_ingestion, IngestionValidationError
from .schemas import SourceType


def discover_course_files(course_root: str | Path) -> List[Path]:
    """
    Discover all valid course files in the directory structure.
    
    Expected structure:
    data/raw/<course>/
      course_notes/
      syllabus/
      lectures/
      notes/
      tutorials/
      exams/
      solutions/
      assignments/
    
    Returns:
        List of file paths that pass validation
    """
    course_root = Path(course_root)
    if not course_root.exists():
        raise ValueError(f"Course root directory does not exist: {course_root}")
    
    valid_files = []
    
    # Files to silently skip (not actual course content)
    skip_patterns = [
        ".gitkeep",
        "README.md",
        ".git",
        "__pycache__",
    ]
    
    # Walk through directory
    for file_path in course_root.rglob("*"):
        if not file_path.is_file():
            continue
        
        # Skip metadata/documentation files silently
        if any(pattern in str(file_path) for pattern in skip_patterns):
            continue
        
        try:
            # Validate file
            validate_file_for_ingestion(file_path, course_root=course_root)
            valid_files.append(file_path)
        except IngestionValidationError as e:
            # Log but continue processing other files
            print(f"Skipping invalid file: {e}")
    
    return valid_files


def ingest_file(
    file_path: str | Path,
    course_root: Optional[str | Path] = None,
    source_type: Optional[SourceType] = None
) -> dict:
    """
    Ingest a single file: validate → parse → chunk → embed.
    
    Args:
        file_path: Path to file to ingest
        course_root: Root directory of course data
        source_type: Optional explicit source_type (will be inferred if not provided)
        
    Returns:
        dict: Metadata about ingested file including chunks and embeddings
        
    Raises:
        IngestionValidationError: If validation fails
    """
    file_path = Path(file_path)
    
    # Validate file format and source_type
    validated_source_type = validate_file_for_ingestion(
        file_path, 
        course_root=course_root,
        source_type=source_type
    )
    
    # TODO: Implement parsing, chunking, and embedding
    # - Parse based on file format (PDF, PPTX, DOCX, MD/TXT)
    # - Chunk based on source_type (different strategies)
    # - Generate embeddings
    # - Store in vector database
    
    return {
        "file_path": str(file_path),
        "source_type": validated_source_type.value,
        "status": "validated",
        # "chunks": [...],
        # "embeddings": [...],
    }


def ingest_course(course_code: str, data_root: str | Path = "data/raw") -> dict:
    """
    Ingest an entire course from data/raw/<course_code>/.
    
    Args:
        course_code: Course code (e.g., "CS240")
        data_root: Root directory containing course directories
        
    Returns:
        dict: Summary of ingestion results
    """
    course_root = Path(data_root) / course_code
    
    if not course_root.exists():
        raise ValueError(f"Course directory does not exist: {course_root}")
    
    # Discover all valid files
    files = discover_course_files(course_root)
    
    if not files:
        print(f"WARNING: No valid files found in {course_root}")
        return {
            "course_code": course_code,
            "files_processed": 0,
            "files_total": 0,
            "results": [],
        }
    
    # Ingest each file
    results = []
    for file_path in files:
        try:
            result = ingest_file(file_path, course_root=course_root)
            results.append(result)
        except IngestionValidationError as e:
            print(f"Failed to ingest {file_path}: {e}")
    
    return {
        "course_code": course_code,
        "files_processed": len(results),
        "files_total": len(files),
        "results": results,
    }
