"""
PDFs → chunks → embeddings

Ingestion pipeline with strict format and source_type validation.
"""
from pathlib import Path
from typing import List, Optional

from .chunking import chunk_by_source_type
from .parsing import parse_file
from .validation import validate_file_for_ingestion, IngestionValidationError
from .schemas import Chunk, SourceType


def _deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Remove duplicate chunks (chunks with identical text content).
    
    Keeps the first occurrence of each unique chunk.
    """
    seen_texts = set()
    unique_chunks = []
    
    for chunk in chunks:
        # Normalize text for comparison (remove extra whitespace)
        normalized = chunk.text.strip().replace('\n', ' ').replace('  ', ' ')
        
        # Skip if we've seen this exact text before
        if normalized in seen_texts:
            continue
        
        seen_texts.add(normalized)
        unique_chunks.append(chunk)
    
    # Re-index chunks after deduplication
    for idx, chunk in enumerate(unique_chunks):
        chunk.chunk_index = idx
    
    return unique_chunks


def _get_chunking_method(source_type: SourceType, parsed_content: dict) -> str:
    """
    Determine which chunking method was used based on source_type and parsed content.
    
    Returns a human-readable description of the chunking strategy.
    """
    if source_type == SourceType.LECTURE_SLIDES:
        if parsed_content.get("slides"):
            return "by_slides"
        return "by_paragraphs (fallback)"
    
    elif source_type in (SourceType.PRACTICE_PROBLEMS, SourceType.EXAM, SourceType.ASSIGNMENT):
        return "by_problems"
    
    elif source_type == SourceType.COURSE_NOTES:
        if parsed_content.get("sections"):
            return "by_sections"
        return "by_paragraphs"
    
    elif source_type in (SourceType.STUDENT_NOTES, SourceType.SYLLABUS, SourceType.SOLUTION):
        return "by_paragraphs"
    
    else:
        return "by_paragraphs (default)"


def _save_chunks_to_files(
    file_path: Path,
    chunks: List[Chunk],
    course_root: Optional[str | Path] = None
) -> None:
    """
    Save chunks to text files for inspection.
    
    Creates directory: data/processed/<course>/<file_stem>/chunks/
    Saves each chunk as a numbered text file with metadata.
    """
    if not chunks:
        return
    
    # Determine course name from course_root
    if course_root:
        course_root_path = Path(course_root) if isinstance(course_root, str) else course_root
        course_name = course_root_path.name
    else:
        # Try to infer from file_path structure
        parts = file_path.parts
        if 'raw' in parts:
            idx = parts.index('raw')
            if idx + 1 < len(parts):
                course_name = parts[idx + 1]
            else:
                course_name = "unknown"
        else:
            course_name = "unknown"
    
    # Create output directory structure
    file_stem = file_path.stem  # filename without extension
    output_dir = Path("data/processed") / course_name / file_stem / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each chunk to a separate file
    for chunk in chunks:
        chunk_filename = f"chunk_{chunk.chunk_index:04d}.txt"
        chunk_file = output_dir / chunk_filename
        
        # Write chunk content with metadata
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"CHUNK #{chunk.chunk_index} - {chunk.id}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"File: {chunk.file_path}\n")
            f.write(f"Source Type: {chunk.source_type.value}\n")
            f.write(f"Locator: {chunk.locator.to_citation()}\n")
            if chunk.heading:
                f.write(f"Heading: {chunk.heading}\n")
            if chunk.char_start is not None and chunk.char_end is not None:
                f.write(f"Character Range: {chunk.char_start}-{chunk.char_end}\n")
            f.write("\n" + "-" * 80 + "\n")
            f.write("CONTENT:\n")
            f.write("-" * 80 + "\n\n")
            f.write(chunk.text)
            f.write("\n")
    
    # Also save a summary file with all chunk IDs
    summary_file = output_dir / "_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Chunks Summary for: {file_path.name}\n")
        f.write(f"Total Chunks: {len(chunks)}\n")
        f.write(f"Source Type: {chunks[0].source_type.value if chunks else 'unknown'}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        for chunk in chunks:
            f.write(f"Chunk {chunk.chunk_index:04d} ({chunk.id}):\n")
            f.write(f"  Location: {chunk.locator.to_citation()}\n")
            f.write(f"  Text Preview: {chunk.text[:100]}...\n")
            f.write(f"  File: chunk_{chunk.chunk_index:04d}.txt\n")
            f.write("\n")


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
    
    # Parse document content
    try:
        parsed_content = parse_file(file_path)
    except Exception as e:
        raise ValueError(f"Error parsing file {file_path}: {e}")
    
    # Extract page numbers for PDF files
    pages = parsed_content["pages"]
    
    # Chunk content based on source_type
    # Note: For PDF files, page numbers are tracked per line,
    # but chunking happens at paragraph/section level.
    # We'll use the first page number in the chunk's text range.
    chunks = chunk_by_source_type(
        text=parsed_content["text"],
        file_path=str(file_path),
        source_type=validated_source_type,
        page=None,  # Page numbers will be set per-chunk if available
        slides=parsed_content["slides"],
        sections=parsed_content["sections"],
    )
    
    # Determine chunking method used
    chunking_method = _get_chunking_method(validated_source_type, parsed_content)
    
    # Deduplicate chunks - remove chunks with identical text content
    chunks = _deduplicate_chunks(chunks)
    
    # Update chunks with page numbers if available (for PDF)
    # Note: We need to find where each chunk's text appears in the original text
    # to correctly assign page numbers and character ranges
    if pages and chunks:
        original_text = parsed_content["text"]
        lines = original_text.split('\n')
        
        for chunk in chunks:
            # Find the position of this chunk's text in the original text
            # Search for the first occurrence of a significant portion of the chunk
            chunk_text = chunk.text
            # Use first 50 chars as a search key (long enough to be unique)
            search_key = chunk_text[:50] if len(chunk_text) >= 50 else chunk_text
            
            # Find position in original text
            char_start = original_text.find(search_key)
            if char_start == -1:
                # If exact match not found, try normalized version
                normalized_original = original_text.replace('\n', ' ').replace('  ', ' ')
                normalized_chunk = chunk_text.replace('\n', ' ').replace('  ', ' ')
                search_key_norm = normalized_chunk[:50] if len(normalized_chunk) >= 50 else normalized_chunk
                char_start = normalized_original.find(search_key_norm)
                if char_start == -1:
                    # Still not found, skip character range assignment
                    continue
            
            char_end = char_start + len(chunk_text)
            
            # Find which page this chunk starts on
            line_start = 0
            for i, line in enumerate(lines):
                line_end = line_start + len(line) + 1  # +1 for newline
                if line_start <= char_start < line_end:
                    if i < len(pages) and pages[i] is not None:
                        chunk.locator.page = pages[i]
                    break
                line_start = line_end
            
            # Update char positions
            chunk.char_start = char_start
            chunk.char_end = char_end
    
    # Save chunks to text files for inspection
    _save_chunks_to_files(file_path, chunks, course_root)
    
    # TODO: Generate embeddings
    # TODO: Store in vector database
    
    return {
        "file_path": str(file_path),
        "source_type": validated_source_type.value,
        "status": "chunked",
        "chunks": [chunk.to_dict() for chunk in chunks],
        "chunk_count": len(chunks),
        "chunking_method": chunking_method,
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
            chunk_count = result.get("chunk_count", 0)
            chunking_method = result.get("chunking_method", "unknown")
            file_name = Path(file_path).name
            print(f"  {file_name}: {chunk_count} chunks ({chunking_method})")
            results.append(result)
        except IngestionValidationError as e:
            print(f"Failed to ingest {file_path}: {e}")
    
    return {
        "course_code": course_code,
        "files_processed": len(results),
        "files_total": len(files),
        "results": results,
    }
