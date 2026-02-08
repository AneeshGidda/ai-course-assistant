#!/usr/bin/env python3
"""
Temporary test script for retrieval functionality.

Run this to test the retrieval system:
    python scripts/test_retrieval.py
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.retrieve import retrieve_chunks, retrieve_chunks_by_course, SourceType


def test_basic_retrieval():
    """Test basic retrieval without filters."""
    print("=" * 80)
    print("TEST 1: Basic Retrieval")
    print("=" * 80)
    
    query = "neural networks"
    print(f"\nQuery: '{query}'")
    print(f"Limit: 5\n")
    
    results = retrieve_chunks(query, limit=5)
    
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Similarity Score: {result.similarity_score:.4f}")
        print(f"  Source Type: {result.chunk.source_type.value}")
        print(f"  File: {Path(result.chunk.file_path).name}")
        print(f"  Citation: {result.chunk.to_citation()}")
        print(f"  Text Preview: {result.chunk.text[:150]}...")
        print()


def test_course_filter():
    """Test retrieval with course filter."""
    print("=" * 80)
    print("TEST 2: Course Filter (CS479)")
    print("=" * 80)
    
    query = "backpropagation"
    course_code = "CS479"
    print(f"\nQuery: '{query}'")
    print(f"Course: {course_code}")
    print(f"Limit: 5\n")
    
    results = retrieve_chunks(query, course_code=course_code, limit=5)
    
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Similarity Score: {result.similarity_score:.4f}")
        print(f"  Source Type: {result.chunk.source_type.value}")
        print(f"  File: {Path(result.chunk.file_path).name}")
        print(f"  Text Preview: {result.chunk.text[:150]}...")
        print()


def test_source_type_filter():
    """Test retrieval with source type filter."""
    print("=" * 80)
    print("TEST 3: Source Type Filter (Lecture Slides Only)")
    print("=" * 80)
    
    query = "gradient descent"
    print(f"\nQuery: '{query}'")
    print(f"Source Types: [lecture_slides]")
    print(f"Limit: 5\n")
    
    results = retrieve_chunks(
        query,
        source_types=[SourceType.LECTURE_SLIDES],
        limit=5
    )
    
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Similarity Score: {result.similarity_score:.4f}")
        print(f"  Source Type: {result.chunk.source_type.value}")
        print(f"  File: {Path(result.chunk.file_path).name}")
        print(f"  Text Preview: {result.chunk.text[:150]}...")
        print()


def test_combined_filters():
    """Test retrieval with both course and source type filters."""
    print("=" * 80)
    print("TEST 4: Combined Filters (CS479 + Lecture Slides)")
    print("=" * 80)
    
    query = "activation functions"
    course_code = "CS479"
    print(f"\nQuery: '{query}'")
    print(f"Course: {course_code}")
    print(f"Source Types: [lecture_slides, course_notes]")
    print(f"Limit: 5\n")
    
    results = retrieve_chunks(
        query,
        course_code=course_code,
        source_types=[SourceType.LECTURE_SLIDES, SourceType.COURSE_NOTES],
        limit=5
    )
    
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Similarity Score: {result.similarity_score:.4f}")
        print(f"  Source Type: {result.chunk.source_type.value}")
        print(f"  File: {Path(result.chunk.file_path).name}")
        print(f"  Citation: {result.chunk.to_citation()}")
        print(f"  Text Preview: {result.chunk.text[:150]}...")
        print()


def test_convenience_function():
    """Test the convenience function for course-specific queries."""
    print("=" * 80)
    print("TEST 5: Convenience Function (retrieve_chunks_by_course)")
    print("=" * 80)
    
    query = "loss function"
    course_code = "CS479"
    print(f"\nQuery: '{query}'")
    print(f"Course: {course_code}")
    print(f"Limit: 3\n")
    
    results = retrieve_chunks_by_course(query, course_code, limit=3)
    
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Similarity Score: {result.similarity_score:.4f}")
        print(f"  Source Type: {result.chunk.source_type.value}")
        print(f"  File: {Path(result.chunk.file_path).name}")
        print(f"  Text Preview: {result.chunk.text[:150]}...")
        print()


def test_min_similarity():
    """Test retrieval with minimum similarity threshold."""
    print("=" * 80)
    print("TEST 6: Minimum Similarity Threshold (0.7)")
    print("=" * 80)
    
    query = "neural networks"
    print(f"\nQuery: '{query}'")
    print(f"Min Similarity: 0.7")
    print(f"Limit: 10\n")
    
    results = retrieve_chunks(query, min_similarity=0.7, limit=10)
    
    print(f"Found {len(results)} results (filtered by min_similarity=0.7)\n")
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Similarity Score: {result.similarity_score:.4f}")
            print(f"  Source Type: {result.chunk.source_type.value}")
            print(f"  File: {Path(result.chunk.file_path).name}")
            print()
    else:
        print("No results above similarity threshold of 0.7")
        print("Try lowering min_similarity or using a more specific query")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RETRIEVAL SYSTEM TEST SUITE")
    print("=" * 80)
    print()
    
    try:
        test_basic_retrieval()
        test_course_filter()
        test_source_type_filter()
        test_combined_filters()
        test_convenience_function()
        test_min_similarity()
        
        print("=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
