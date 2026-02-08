#!/usr/bin/env python3
"""
Test script for RAG answer generation with citations.

Run this to test the answer generation system:
    python scripts/test_rag_generation.py
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag.generate import generate_answer, generate_answer_with_evidence
from app.rag.schemas import SourceType
from app.rag.exceptions import InsufficientMaterialError


def test_basic_answer():
    """Test basic answer generation."""
    print("=" * 80)
    print("TEST 1: Basic Answer Generation")
    print("=" * 80)
    
    query = "What is backpropagation?"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479\n")
    
    try:
        result = generate_answer(
            query=query,
            course_code="CS479",
            retrieval_limit=5,
        )
        
        print("Answer:")
        print("-" * 80)
        print(result.answer)
        print("-" * 80)
        
        print(f"\nCitations found: {len(result.citations)}")
        for citation in result.citations:
            print(f"  - {citation}")
        
        print(f"\nChunks used: {len(result.chunks_used)}")
        print(f"Evidence chunks: {len(result.retrieval_results)}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_teaching_mode():
    """Test teaching mode (more explanatory)."""
    print("=" * 80)
    print("TEST 2: Teaching Mode (Explanatory)")
    print("=" * 80)
    
    query = "How does gradient descent work?"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479")
    print(f"Mode: teach\n")
    
    try:
        result = generate_answer(
            query=query,
            course_code="CS479",
            retrieval_limit=5,
            mode="teach",
        )
        
        print("Answer:")
        print("-" * 80)
        print(result.answer)
        print("-" * 80)
        
        print(f"\nCitations: {len(result.citations)}")
        for citation in result.citations:
            print(f"  - {citation}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_source_type_filter():
    """Test answer generation with source type filter."""
    print("=" * 80)
    print("TEST 3: Answer with Source Type Filter (Lecture Slides Only)")
    print("=" * 80)
    
    query = "What are activation functions?"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479")
    print(f"Source Types: [lecture_slides]\n")
    
    try:
        result = generate_answer(
            query=query,
            course_code="CS479",
            source_types=[SourceType.LECTURE_SLIDES],
            retrieval_limit=5,
        )
        
        print("Answer:")
        print("-" * 80)
        print(result.answer)
        print("-" * 80)
        
        print(f"\nCitations: {len(result.citations)}")
        for citation in result.citations:
            print(f"  - {citation}")
        
        print(f"\nEvidence chunks:")
        for i, evidence in enumerate(result.retrieval_results[:3], 1):
            print(f"  {i}. {evidence.chunk.to_citation()} (similarity: {evidence.similarity_score:.3f})")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_citation_validation():
    """Test that citations are properly extracted and validated."""
    print("=" * 80)
    print("TEST 4: Citation Validation")
    print("=" * 80)
    
    query = "Explain neural networks"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479\n")
    
    try:
        result = generate_answer(
            query=query,
            course_code="CS479",
            retrieval_limit=5,
        )
        
        print("Answer (showing citations):")
        print("-" * 80)
        # Highlight citations in the answer
        answer_with_highlights = result.answer
        import re
        citation_pattern = r'(\[Citation:[^\]]+\])'
        answer_with_highlights = re.sub(
            citation_pattern,
            r'>>>\1<<<',
            answer_with_highlights
        )
        print(answer_with_highlights)
        print("-" * 80)
        
        print(f"\nExtracted Citations ({len(result.citations)}):")
        for citation in result.citations:
            print(f"  ✓ {citation}")
        
        print(f"\nEvidence Chunks ({len(result.retrieval_results)}):")
        for i, evidence in enumerate(result.retrieval_results, 1):
            chunk_citation = evidence.chunk.to_citation()
            print(f"  {i}. {chunk_citation}")
            print(f"     Similarity: {evidence.similarity_score:.3f}")
            print(f"     Preview: {evidence.chunk.text[:100]}...")
            print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_convenience_function():
    """Test the convenience function that returns a dictionary."""
    print("=" * 80)
    print("TEST 5: Convenience Function (Dictionary Output)")
    print("=" * 80)
    
    query = "What is a loss function?"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479\n")
    
    try:
        result_dict = generate_answer_with_evidence(
            query=query,
            course_code="CS479",
            retrieval_limit=5,
        )
        
        print("Answer:")
        print("-" * 80)
        print(result_dict["answer"])
        print("-" * 80)
        
        print(f"\nCitations: {result_dict['citations']}")
        print(f"Chunks Used: {len(result_dict['chunks_used'])}")
        
        print(f"\nEvidence ({len(result_dict['evidence'])} chunks):")
        for i, evidence in enumerate(result_dict["evidence"][:3], 1):
            print(f"  {i}. {evidence['citation']}")
            print(f"     Similarity: {evidence['similarity_score']:.3f}")
            print(f"     Preview: {evidence['text_preview']}...")
            print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_no_citations_warning():
    """Test handling when LLM doesn't include citations."""
    print("=" * 80)
    print("TEST 6: Answer Generation (Check for Citations)")
    print("=" * 80)
    
    query = "What are the main topics in this course?"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479\n")
    
    try:
        result = generate_answer(
            query=query,
            course_code="CS479",
            retrieval_limit=5,
        )
        
        print("Answer:")
        print("-" * 80)
        print(result.answer)
        print("-" * 80)
        
        # Check if answer contains citations
        import re
        citation_pattern = r'\[Citation:[^\]]+\]'
        citations_in_text = re.findall(citation_pattern, result.answer, re.IGNORECASE)
        
        print(f"\nCitations in answer text: {len(citations_in_text)}")
        if citations_in_text:
            print("  Citations found:")
            for citation in citations_in_text:
                print(f"    - {citation}")
        else:
            print("  ⚠ WARNING: No citations found in answer text!")
            print("  The LLM may not have followed citation requirements.")
        
        print(f"\nExtracted citations: {len(result.citations)}")
        print(f"Evidence chunks available: {len(result.retrieval_results)}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_refusal_behavior():
    """Test that system refuses to answer when retrieval quality is too low."""
    print("=" * 80)
    print("TEST 7: Refusal Behavior (Low Retrieval Quality)")
    print("=" * 80)
    
    # Use a query that likely won't match well (e.g., completely unrelated topic)
    query = "What is the capital of Mars and how do quantum computers work in zero gravity?"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479")
    print(f"Expected: System should refuse to answer due to low similarity\n")
    
    try:
        result = generate_answer(
            query=query,
            course_code="CS479",
            retrieval_limit=5,
            strict_quality_check=True,  # Enable strict quality checking
        )
        
        # If we get here, the system answered (might happen if threshold is very low)
        print("⚠ WARNING: System generated an answer despite low quality query.")
        print("This might indicate the quality threshold is too low.")
        print("\nAnswer:")
        print("-" * 80)
        print(result.answer)
        print("-" * 80)
        
        if result.retrieval_quality:
            print(f"\nRetrieval Quality Metrics:")
            print(f"  Top Similarity: {result.retrieval_quality['top_similarity']:.3f}")
            print(f"  High Quality Chunks: {result.retrieval_quality['high_quality_chunks']}")
            print(f"  Total Chunks: {result.retrieval_quality['total_chunks']}")
            print(f"  Is Sufficient: {result.retrieval_quality['is_sufficient']}")
        
    except InsufficientMaterialError as e:
        print("✓ CORRECT: System refused to answer due to insufficient material")
        print("\nRefusal Message:")
        print("-" * 80)
        print(e.message)
        print("-" * 80)
        
        print(f"\nQuality Metrics:")
        print(f"  Top Similarity: {e.top_similarity:.3f}")
        print(f"  High Quality Chunks: {e.high_quality_chunks}")
        print(f"  Total Chunks: {e.total_chunks}")
        print(f"\nThis demonstrates the hallucination prevention system working correctly.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_quality_metrics():
    """Test that quality metrics are included in successful answers."""
    print("=" * 80)
    print("TEST 8: Quality Metrics in Successful Answers")
    print("=" * 80)
    
    query = "What is backpropagation?"
    print(f"\nQuery: '{query}'")
    print(f"Course: CS479\n")
    
    try:
        result = generate_answer(
            query=query,
            course_code="CS479",
            retrieval_limit=5,
            strict_quality_check=True,
        )
        
        print("Answer:")
        print("-" * 80)
        print(result.answer[:200] + "..." if len(result.answer) > 200 else result.answer)
        print("-" * 80)
        
        if result.retrieval_quality:
            print(f"\n✓ Quality Metrics Included:")
            print(f"  Top Similarity: {result.retrieval_quality['top_similarity']:.3f}")
            print(f"  High Quality Chunks: {result.retrieval_quality['high_quality_chunks']}")
            print(f"  Total Chunks: {result.retrieval_quality['total_chunks']}")
            print(f"  Min Confidence Threshold: {result.retrieval_quality['min_confidence_threshold']:.3f}")
            print(f"  Min High Quality Chunks Required: {result.retrieval_quality['min_high_quality_chunks_required']}")
            print(f"  Is Sufficient: {result.retrieval_quality['is_sufficient']}")
        else:
            print("\n⚠ WARNING: No quality metrics found in result")
        
    except InsufficientMaterialError as e:
        print(f"⚠ System refused to answer: {e.message}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RAG ANSWER GENERATION TEST SUITE")
    print("=" * 80)
    print()
    print("This will test:")
    print("  - Answer generation with citations")
    print("  - Teaching mode")
    print("  - Source type filtering")
    print("  - Citation extraction and validation")
    print("  - Evidence tracking")
    print("  - Hallucination prevention (refusal behavior)")
    print("  - Quality metrics")
    print()
    
    try:
        test_basic_answer()
        test_teaching_mode()
        test_source_type_filter()
        test_citation_validation()
        test_convenience_function()
        test_no_citations_warning()
        test_refusal_behavior()
        test_quality_metrics()
        
        print("=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nNote: Check that answers include [Citation: ...] markers")
        print("      and that citations reference actual course materials.")
        print("      Low-quality queries should be refused to prevent hallucinations.")
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
