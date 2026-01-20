# File Ingestion Guide

This document defines the strict file format and directory structure requirements for ingesting course materials.

## 1. Allowed File Formats

**✅ Supported formats (v1):**

| Format | Why |
|--------|-----|
| `.pdf` | slides, notes, exams, syllabus |
| `.pptx` | lecture slides (best structured) |
| `.docx` | notes, worksheets |
| `.md` / `.txt` | clean notes |

**❌ Everything else is rejected** or must be converted to PDF first.

This keeps ingestion deterministic and debuggable.

## 2. Source Type Enum

The semantic `source_type` is **NOT inferred from content**. It's a fixed enum you explicitly control.

**✅ Core source_type values (v1):**

- `course_notes` - Official course notes
- `lecture_slides` - Lecture presentations
- `student_notes` - Student-generated notes
- `syllabus` - Course syllabus
- `practice_problems` - Practice/tutorial problems
- `exam` - Exam papers
- `solution` - Solutions to problems/exams
- `assignment` - Assignment specifications

Each source type has:
- **Different chunking** strategies
- **Different citation styles**
- **Different retrieval priority**

## 3. Format → Source Type Mappings

Not every format can be every type. These are the **allowed combinations**:

| source_type | allowed formats |
|-------------|----------------|
| `course_notes` | pdf, docx, md |
| `lecture_slides` | pptx, pdf |
| `student_notes` | docx, md, pdf |
| `syllabus` | pdf, docx |
| `practice_problems` | pdf, docx |
| `exam` | pdf |
| `solution` | pdf, docx |
| `assignment` | pdf, docx |

If a file violates this mapping, **ingestion will fail loudly**.

## 4. Directory Structure Contract

This is how `source_type` is **deterministically determined**.

Expected structure:
```
data/raw/<course>/
  course_notes/
    *.pdf
    *.docx
    *.md
  syllabus/
    *.pdf
    *.docx
  lectures/
    *.pptx
    *.pdf
  notes/
    *.docx
    *.md
    *.pdf
  tutorials/
    *.pdf
    *.docx
  exams/
    *.pdf
  solutions/
    *.pdf
    *.docx
  assignments/
    *.pdf
    *.docx
```

**Directory → source_type mapping:**

| Directory | source_type |
|-----------|-------------|
| `course_notes/` | `course_notes` |
| `syllabus/` | `syllabus` |
| `lectures/` | `lecture_slides` |
| `notes/` | `student_notes` |
| `tutorials/` | `practice_problems` |
| `exams/` | `exam` |
| `solutions/` | `solution` |
| `assignments/` | `assignment` |

## Example: Valid Course Structure

```
data/raw/CS240/
  course_notes/
    week1.pdf
    chapter1.docx
  syllabus/
    syllabus.pdf
    outline.docx
  lectures/
    lecture01.pptx
    lecture02.pdf
  notes/
    week1-notes.md
    week2-notes.docx
  tutorials/
    tutorial1.pdf
    tutorial2.docx
  exams/
    midterm2023.pdf
    final2023.pdf
  solutions/
    tutorial1-solution.pdf
    midterm-solution.pdf
  assignments/
    a1.pdf
    a2.pdf
```

## Usage

Ingest a course:
```bash
python scripts/ingest_course.py CS240
```

The ingestion system will:
1. ✅ Validate file formats (must be in allowed list)
2. ✅ Infer source_type from directory structure
3. ✅ Validate format → source_type mapping
4. ❌ **Fail loudly** if any validation step fails

## Validation Errors

If you see errors like:
- `File format '.xlsx' is not supported` → Convert to PDF or use an allowed format
- `File format '.pptx' is not allowed for source_type 'exam'` → Move file to correct directory
- `Cannot infer source_type from path` → Check directory structure matches expected format
