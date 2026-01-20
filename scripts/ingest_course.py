#!/usr/bin/env python3
"""
CLI: ingest a course

Usage:
    python scripts/ingest_course.py CS240
    python scripts/ingest_course.py CS240 --data-root data/raw
"""
import sys
import argparse
import traceback
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.ingest import ingest_course
from app.rag.validation import IngestionValidationError


def main():
    parser = argparse.ArgumentParser(
        description="Ingest course materials with strict format and source_type validation"
    )
    parser.add_argument(
        "course_code",
        type=str,
        help="Course code (e.g., CS240)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root directory containing course directories (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Ingesting course: {args.course_code}")
        print(f"Data root: {args.data_root}")
        print()
        
        result = ingest_course(args.course_code, data_root=args.data_root)
        
        print(f"\nIngestion complete!")
        print(f"   Files processed: {result['files_processed']}/{result['files_total']}")
        
        if result['files_processed'] == 0:
            print("\nWARNING: No files were processed. Check:")
            print("   1. Directory structure matches expected format")
            print("   2. File formats are in allowed list (.pdf, .pptx, .docx, .md, .txt)")
            print("   3. Format → source_type mappings are valid")
            sys.exit(1)
        
        # Print summary by source_type
        by_type = {}
        for r in result['results']:
            st = r['source_type']
            by_type[st] = by_type.get(st, 0) + 1
        
        print("\nFiles by source_type:")
        for st, count in sorted(by_type.items()):
            print(f"   {st}: {count}")
        
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except IngestionValidationError as e:
        print(f"ERROR: Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
