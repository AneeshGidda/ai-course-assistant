"""
File format and source type validation for ingestion.

Enforces strict format → source_type mappings and directory contracts.
"""
import os
from pathlib import Path
from typing import Optional

from .schemas import SourceType, FORMAT_TO_SOURCE_TYPE, ALLOWED_FORMATS


class IngestionValidationError(Exception):
    """Raised when file format or source type validation fails."""
    pass


def get_file_extension(file_path: str | Path) -> str:
    """Extract file extension (lowercase, including dot)."""
    return Path(file_path).suffix.lower()


def is_allowed_format(file_path: str | Path) -> bool:
    """Check if file format is in the allowed list."""
    ext = get_file_extension(file_path)
    return ext in ALLOWED_FORMATS


def validate_format(file_path: str | Path) -> None:
    """
    Validate file format. Raises IngestionValidationError if not allowed.
    
    Args:
        file_path: Path to file to validate
        
    Raises:
        IngestionValidationError: If format is not in allowed list
    """
    ext = get_file_extension(file_path)
    if ext not in ALLOWED_FORMATS:
        raise IngestionValidationError(
            f"File format '{ext}' is not supported. "
            f"Allowed formats: {', '.join(ALLOWED_FORMATS.keys())}. "
            f"File: {file_path}"
        )


def validate_format_source_type_mapping(file_path: str | Path, source_type: SourceType) -> None:
    """
    Validate that the file format is allowed for the given source_type.
    
    Args:
        file_path: Path to file
        source_type: Intended source type
        
    Raises:
        IngestionValidationError: If format is not allowed for this source_type
    """
    validate_format(file_path)  # First check if format is allowed at all
    ext = get_file_extension(file_path)
    allowed_types = FORMAT_TO_SOURCE_TYPE.get(ext, set())
    
    if source_type not in allowed_types:
        raise IngestionValidationError(
            f"File format '{ext}' is not allowed for source_type '{source_type.value}'. "
            f"Allowed source_types for '{ext}': {[t.value for t in allowed_types]}. "
            f"File: {file_path}"
        )


def infer_source_type_from_path(file_path: str | Path, course_root: Optional[str | Path] = None) -> SourceType:
    """
    Infer source_type from directory structure.
    
    Directory → source_type mapping:
    - course_notes/ -> course_notes
    - syllabus/ -> syllabus
    - lectures/ -> lecture_slides
    - notes/ -> student_notes
    - tutorials/ -> practice_problems
    - exams/ -> exam
    - solutions/ -> solution
    - assignments/ -> assignment
    
    Args:
        file_path: Path to file (absolute or relative to course_root)
        course_root: Root directory of course data (e.g., data/raw/<course>/)
        
    Returns:
        SourceType: Inferred source type
        
    Raises:
        IngestionValidationError: If source_type cannot be determined
    """
    file_path = Path(file_path)
    
    # If course_root is provided, make path relative to it
    if course_root:
        try:
            rel_path = file_path.relative_to(Path(course_root))
        except ValueError:
            # File is outside course_root, use absolute path
            rel_path = file_path
    else:
        rel_path = file_path
    
    # Get directory name
    parent_dir = rel_path.parent.name.lower()
    
    # Check directory patterns
    if parent_dir in ("course_notes", "course-notes"):
        return SourceType.COURSE_NOTES
    
    if parent_dir in ("syllabus", "syllabi"):
        return SourceType.SYLLABUS
    
    if parent_dir in ("lectures", "lecture"):
        return SourceType.LECTURE_SLIDES
    
    if parent_dir in ("notes", "note"):
        return SourceType.STUDENT_NOTES
    
    if parent_dir in ("tutorials", "tutorial", "practice", "practices"):
        return SourceType.PRACTICE_PROBLEMS
    
    if parent_dir in ("exams", "exam", "tests", "test"):
        return SourceType.EXAM
    
    if parent_dir in ("solutions", "solution", "sol", "sols"):
        return SourceType.SOLUTION
    
    if parent_dir in ("assignments", "assignment", "hw", "homework"):
        return SourceType.ASSIGNMENT
    
    # If we can't determine, raise error
    raise IngestionValidationError(
        f"Cannot infer source_type from path: {rel_path}. "
        f"Expected directory structure: data/raw/<course>/{{course_notes/, syllabus/, lectures/, notes/, tutorials/, exams/, solutions/, assignments/}}. "
        f"File: {file_path}"
    )


def validate_file_for_ingestion(
    file_path: str | Path,
    course_root: Optional[str | Path] = None,
    source_type: Optional[SourceType] = None
) -> SourceType:
    """
    Complete validation: format, source_type inference, and format→source_type mapping.
    
    Args:
        file_path: Path to file to validate
        course_root: Root directory of course data
        source_type: Optional explicit source_type (if not provided, will be inferred)
        
    Returns:
        SourceType: Validated source type
        
    Raises:
        IngestionValidationError: If validation fails at any step
    """
    # Step 1: Validate format
    validate_format(file_path)
    
    # Step 2: Infer or use provided source_type
    if source_type is None:
        source_type = infer_source_type_from_path(file_path, course_root)
    
    # Step 3: Validate format → source_type mapping
    validate_format_source_type_mapping(file_path, source_type)
    
    return source_type
