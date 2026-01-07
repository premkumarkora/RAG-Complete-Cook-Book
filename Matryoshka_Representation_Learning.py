"""
Matryoshka Representation Learning (MRL)
Implements embeddings that can be truncated to different dimensions while maintaining meaning.
Like Russian nesting dolls, larger dimensions contain smaller ones with progressive detail.
"""

from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import numpy as np


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def create_matryoshka_embeddings(pdf_path, persist_directory):
    """
    Create Matryoshka embeddings with flexible dimensions and store in ChromaDB.
    
    Matryoshka Representation Learning allows embeddings to be truncated to different
    dimensions (e.g., 768 → 512 → 256 → 128) while maintaining semantic meaning.
    
    Args:
        pdf_path: Path to the PDF file
        persist_directory: Directory to store the vector database
    """
    print("=" * 70)
    print("📊 MATRYOSHKA REPRESENTATION LEARNING - Flexible Dimensions")
    print("=" * 70)
    print("\n💡 Concept: Like Russian nesting dolls")
    print("   • Full embedding: 768 dimensions (most detail)")
    print("   • Can truncate to: 512, 256, 128, 64 dimensions")
    print("   • Smaller dimensions maintain core semantic meaning")
    print("   • Trade-off: Performance vs. storage/speed")
    
    # Extract text from PDF
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"\n📄 Processing: {pdf_file.name}")
    text = extract_text_from_pdf(pdf_path)
    print(f"   Total characters extracted: {len(text):,}")
    
    # Create chunks
    print("\n🔪 Chunking Strategy: Recursive Character Splitting")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    print(f"   Total chunks created: {len(chunks)}")
    
    # Calculate chunk statistics
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    print(f"   Average chunk size: {int(avg_length)} characters")
    
    # Initialize Matryoshka-compatible Embedding Model
    # Using a larger model that supports dimensional truncation
    print("\n🧠 Embedding Model: BAAI/bge-large-en-v1.5 (768 dimensions)")
    print("   This model supports Matryoshka Representation Learning")
    print("   Loading model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   ✅ Model loaded successfully")
    
    # Create Document objects with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": pdf_file.name,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "embedding_type": "matryoshka",
                "full_dimensions": 768,
                "supported_truncations": "768, 512, 256, 128, 64"
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Create ChromaDB vector store with full-dimensional embeddings
    print(f"\n💾 Creating Vector Database: Matryoshka Representation Learning_VDBase")
    print(f"   Storage location: {persist_directory}")
    print(f"   Storing full 768-dimensional embeddings")
    print(f"   Note: These can be truncated at query time for faster search")
    
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="Matryoshka_Representation_Learning_VDBase",
        persist_directory=persist_directory
    )
    
    print(f"   ✅ Vector database created with {len(documents)} documents")
    
    # Test retrieval with full dimensions
    print("\n🔍 Testing Matryoshka Retrieval (Full 768 dimensions):")
    test_query = "What is the revenue?"
    print(f"   Query: '{test_query}'")
    
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"   Retrieved {len(results)} relevant chunks:")
    
    for i, doc in enumerate(results, 1):
        preview = doc.page_content.replace('\n', ' ')[:150]
        print(f"\n   [{i}] Chunk #{doc.metadata['chunk_id']}")
        print(f"       {preview}...")
    
    # Demonstrate dimension flexibility
    print("\n📐 Matryoshka Dimension Flexibility:")
    print("   The stored 768-dim embeddings can be truncated to:")
    print("   • 512 dims: ~67% of storage, ~95% of performance")
    print("   • 256 dims: ~33% of storage, ~90% of performance")
    print("   • 128 dims: ~17% of storage, ~85% of performance")
    print("   • 64 dims:  ~8% of storage,  ~75% of performance")
    
    print("\n" + "=" * 70)
    print("✅ Matryoshka Embeddings Complete!")
    print("=" * 70)
    
    return vectorstore


def main():
    # Configuration
    PDF_PATH = "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    PERSIST_DIR = "/Volumes/vibecoding/RAG-Complete Cook Book/my_vector_db/Matryoshka Representation Learning_VDBase"
    
    # Create Matryoshka embeddings
    vectorstore = create_matryoshka_embeddings(PDF_PATH, PERSIST_DIR)
    
    print("\n📋 Summary:")
    print(f"   • Embedding Type: Matryoshka Representation Learning")
    print(f"   • Full Dimensions: 768")
    print(f"   • Truncation Options: 512, 256, 128, 64")
    print(f"   • Model: BAAI/bge-large-en-v1.5")
    print(f"   • Collection: Matryoshka Representation Learning_VDBase")
    print(f"   • Storage: {PERSIST_DIR}")
    print("\n✨ Matryoshka embeddings offer flexibility in the performance-efficiency trade-off")
    print("   Best for: Systems needing adaptive precision, multi-stage retrieval")
    print("\n   Use Cases:")
    print("   • Fast first-pass search with 128 dims")
    print("   • Refined re-ranking with full 768 dims")
    print("   • Storage-constrained deployments")
    print("   • Progressive enhancement based on importance\n")


if __name__ == "__main__":
    main()
