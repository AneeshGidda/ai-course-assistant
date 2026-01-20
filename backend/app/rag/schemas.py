"""
Structured LLM outputs and data schemas for RAG ingestion.
"""
from enum import Enum
from typing import Literal


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
