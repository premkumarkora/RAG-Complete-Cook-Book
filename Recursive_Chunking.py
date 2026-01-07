"""
Recursive Chunking - The Recommended Default
Splits intelligently: Paragraphs → Lines → Sentences → Words
"""

from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def main():
    # PDF files
    pdfs = [
        "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-August-Q1-2526.pdf",
        "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    ]
    
    # Configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 75  # 15% of 500
    
    # Initialize Recursive Splitter
    # It will try separators in this order: ["\n\n", "\n", ".", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # Paragraph → Line → Sentence → Word
        length_function=len
    )
    
    print(f"📄 Recursive Chunking (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    print(f"   Strategy: Paragraph → Line → Sentence → Word")
    print("=" * 70)
    
    for pdf_path in pdfs:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            print(f"❌ File not found: {pdf_file.name}\n")
            continue
        
        print(f"\n📂 Processing: {pdf_file.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        # Create chunks using Recursive splitter
        chunks = text_splitter.split_text(text)
        
        print(f"   Total characters: {len(text):,}")
        print(f"   Total chunks: {len(chunks)}")
        
        # Show first 3 chunks
        print(f"\n   First 3 chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   --- Chunk {i} (Length: {len(chunk)}) ---")
            # Show first 150 chars
            preview = chunk.replace('\n', ' ')
            print(f"   {preview}  ", len(preview))
        
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
