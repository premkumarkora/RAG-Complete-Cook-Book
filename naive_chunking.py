"""
Fixed-Size Chunking with Overlap
Simple and elegant implementation for PDF processing
"""

from pypdf import PdfReader
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap_percent=0.15):
    """
    Split text into fixed-size chunks with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Number of characters per chunk
        overlap_percent: Percentage of overlap (0.15 = 15%)
    
    Returns:
        List of text chunks
    """
    chunks = []
    overlap_size = int(chunk_size * overlap_percent)
    start = 0
    
    while start < len(text):
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move forward by (chunk_size - overlap_size)
        start += chunk_size - overlap_size
        
    return chunks


def main():
    # PDF files
    pdfs = [
        "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-August-Q1-2526.pdf",
        "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    ]
    
    # Configuration
    CHUNK_SIZE = 500
    OVERLAP = 0.15  # 15% overlap
    
    print(f"📄 Fixed-Size Chunking (Size: {CHUNK_SIZE}, Overlap: {int(OVERLAP*100)}%)")
    print("=" * 70)
    
    for pdf_path in pdfs:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            print(f"❌ File not found: {pdf_file.name}\n")
            continue
        
        print(f"\n📂 Processing: {pdf_file.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        # Create chunks
        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        
        print(f"   Total characters: {len(text):,}")
        print(f"   Total chunks: {len(chunks)}")
        
        # Show first 3 chunks
        print(f"\n   First 3 chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   --- Chunk {i} (Length: {len(chunk)}) ---")
            # Show first 150 chars
            preview = chunk[:150].replace('\n', ' ')
            print(f"   {preview}...")
        
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
