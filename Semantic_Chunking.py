"""
Semantic Chunking - Topic-Based Splitting
Uses embeddings to detect when the topic changes
"""

from pypdf import PdfReader
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


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
    
    print("📄 Semantic Chunking - Topic-Based Splitting")
    print("   How it works:")
    print("   1. Splits text into sentences")
    print("   2. Calculates semantic similarity between consecutive sentences")
    print("   3. Creates new chunk when topic shifts (similarity drops)")
    print("=" * 70)
    
    # Initialize Embedding Model
    # Using a lightweight model that runs locally on CPU
    print("\n Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Initialize Semantic Chunker
    # 'percentile' threshold: splits when similarity is in the bottom X%
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    print("✅ Model loaded. Processing PDFs...\n")
    
    for pdf_path in pdfs:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            print(f"❌ File not found: {pdf_file.name}\n")
            continue
        
        print(f"📂 Processing: {pdf_file.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        # Create chunks using Semantic splitter
        # This will analyze sentence-by-sentence and group by topic
        chunks = text_splitter.split_text(text)
        
        print(f"   Total characters: {len(text):,}")
        print(f"   Total chunks: {len(chunks)}")
        
        # Analyze chunk sizes
        chunk_lengths = [len(chunk) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        
        print(f"   Average chunk size: {int(avg_length)} chars")
        print(f"   Smallest chunk: {min(chunk_lengths)} chars")
        print(f"   Largest chunk: {max(chunk_lengths)} chars")
        
        # Show first 3 chunks
        print(f"\n   First 3 chunks (topic-based):")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   --- Chunk {i} (Length: {len(chunk)}) ---")
            # Show first 200 chars to understand the topic
            preview = chunk.replace('\n', ' ')
            print(f"   {preview}  ", len(preview))
        
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()

"""
Key Observations:
Comparison with Previous Methods:

Method	ITC-August Chunks	ITC-October Chunks	Strategy
Naive	46 chunks	44 chunks	Fixed 500 chars
Recursive	44 chunks	43 chunks	Paragraph/sentence boundaries
Semantic	7 chunks	6 chunks	Topic shifts

Why So Few Chunks?

The semantic chunker detected that most of the PDF discusses related financial topics (Revenue, EBITDA, Segment performance)
Instead of arbitrarily cutting every 500 chars, it kept entire topics together
Notice the huge variation in chunk sizes:
Smallest: 33 chars (probably a table or header)
Largest: 8,805 chars (entire discussion about FMCG segment)
Pros: ✅ Each chunk is about one cohesive topic
✅ Perfect for Q&A systems (entire context in one chunk)
✅ No mid-sentence cuts

Cons: ❌ Slower (had to embed every sentence)
❌ Variable chunk sizes (some very large)
"""
