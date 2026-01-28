"""
Document parsing layer using LangChain document loaders.

Extracts text content from various file formats (PDF, PPTX, DOCX, MD, TXT).
Returns raw text and structure information (pages, slides, sections).
"""
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain_core.documents import Document as LangChainDocument

# Suppress pypdf warnings about malformed PDF objects
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")


def parse_text_file(file_path: Path) -> list[LangChainDocument]:
    """Parse plain text or markdown file using LangChain TextLoader."""
    try:
        loader = TextLoader(str(file_path), encoding='utf-8')
        documents = loader.load()
        return documents
    except UnicodeDecodeError:
        # Try alternative encoding
        loader = TextLoader(str(file_path), encoding='latin-1')
        documents = loader.load()
        return documents


def parse_pdf(file_path: Path) -> list[LangChainDocument]:
    """
    Parse PDF file using LangChain PyPDFLoader.
    
    Returns LangChain Document objects with page numbers in metadata.
    """
    try:
        # Suppress pypdf warnings about malformed PDF objects
        # These warnings are harmless and don't affect text extraction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*wrong pointing object.*")
            # Also redirect stderr to suppress pypdf print statements
            with redirect_stderr(io.StringIO()):
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
        # PyPDFLoader automatically adds page numbers to metadata
        return documents
    except Exception as e:
        raise ValueError(f"Error parsing PDF file {file_path}: {e}")


def parse_pptx(file_path: Path) -> list[LangChainDocument]:
    """
    Parse PPTX file using LangChain UnstructuredPowerPointLoader.
    
    Returns LangChain Document objects, one per slide.
    """
    try:
        loader = UnstructuredPowerPointLoader(str(file_path))
        documents = loader.load()
        # Each document represents a slide
        return documents
    except Exception as e:
        raise ValueError(f"Error parsing PPTX file {file_path}: {e}")


def parse_docx(file_path: Path) -> list[LangChainDocument]:
    """
    Parse DOCX file using LangChain UnstructuredWordDocumentLoader.
    
    Returns LangChain Document objects with potential section information.
    """
    try:
        loader = UnstructuredWordDocumentLoader(str(file_path))
        documents = loader.load()
        return documents
    except Exception as e:
        raise ValueError(f"Error parsing DOCX file {file_path}: {e}")


def parse_file(file_path: Path) -> dict:
    """
    Parse file based on its extension using LangChain loaders.
    
    Converts LangChain Document objects to our internal format.
    
    Returns dictionary with:
        - text: Full document text
        - pages: List of page numbers corresponding to lines/paragraphs in text (for PDF)
        - slides: List of slide texts (for PPTX)
        - sections: Dictionary of section_name -> section_text (for DOCX with headings)
        - documents: Original LangChain Document objects for advanced use
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".md" or suffix == ".txt":
        documents = parse_text_file(file_path)
        # Combine all document content
        text = "\n\n".join([doc.page_content for doc in documents])
        
        return {
            "text": text,
            "pages": None,
            "slides": None,
            "sections": None,
            "documents": documents,
        }
    
    elif suffix == ".pdf":
        documents = parse_pdf(file_path)
        
        # Extract text and page numbers
        text_parts = []
        page_numbers = []
        
        for doc in documents:
            page_num = doc.metadata.get("page", None)
            content = doc.page_content
            
            # Split content into lines to track page numbers
            lines = content.split('\n')
            text_parts.extend(lines)
            page_numbers.extend([page_num] * len(lines) if page_num is not None else [None] * len(lines))
        
        full_text = '\n'.join(text_parts)
        
        return {
            "text": full_text,
            "pages": page_numbers,  # List of page numbers for each line
            "slides": None,
            "sections": None,
            "documents": documents,
        }
    
    elif suffix == ".pptx":
        documents = parse_pptx(file_path)
        
        # Extract slide texts
        slides = [doc.page_content for doc in documents]
        
        # Combine all slides into full text for fallback
        full_text = "\n\n".join(slides) if slides else ""
        
        return {
            "text": full_text,
            "pages": None,
            "slides": slides,
            "sections": None,
            "documents": documents,
        }
    
    elif suffix == ".docx":
        documents = parse_docx(file_path)
        
        # Extract text
        text_parts = [doc.page_content for doc in documents]
        full_text = "\n\n".join(text_parts)
        
        # Try to identify sections from metadata
        sections = None
        section_dict = {}
        current_section = None
        current_section_text = []
        
        for doc in documents:
            # Check metadata for section information
            metadata = doc.metadata
            # Some loaders might include heading information
            if "heading" in metadata or "category" in metadata:
                # Save previous section
                if current_section and current_section_text:
                    section_dict[current_section] = "\n".join(current_section_text)
                    current_section_text = []
                
                # Start new section
                current_section = metadata.get("heading") or metadata.get("category")
                current_section_text.append(doc.page_content)
            else:
                if current_section:
                    current_section_text.append(doc.page_content)
        
        # Save last section
        if current_section and current_section_text:
            section_dict[current_section] = "\n".join(current_section_text)
        
        sections = section_dict if section_dict else None
        
        return {
            "text": full_text,
            "pages": None,
            "slides": None,
            "sections": sections,
            "documents": documents,
        }
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
