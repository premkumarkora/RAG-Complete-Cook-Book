"""
Dense Embeddings with ChromaDB
Creates standard dense vector embeddings using HuggingFace's all-MiniLM-L6-v2 model.
Dense embeddings represent text as continuous vectors in high-dimensional space (384 dimensions).
"""

from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def create_dense_embeddings(pdf_path, persist_directory):
    """
    Create dense embeddings and store in ChromaDB.
    
    Args:
        pdf_path: Path to the PDF file
        persist_directory: Directory to store the vector database
    """
    print("=" * 70)
    print("📊 DENSE EMBEDDINGS - Standard Vector Representation")
    print("=" * 70)
    
    # Extract text from PDF
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"\n📄 Processing: {pdf_file.name}")
    text = extract_text_from_pdf(pdf_path)
    print(f"   Total characters extracted: {len(text):,}")
    
    # Create chunks using RecursiveCharacterTextSplitter
    # This splits on natural boundaries: paragraphs → sentences → words
    print("\n🔪 Chunking Strategy: Recursive Character Splitting")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Target chunk size
        chunk_overlap=50,  # Overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Natural split points
    )
    
    chunks = text_splitter.split_text(text)
    print(f"   Total chunks created: {len(chunks)}")
    
    # Calculate chunk statistics
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    print(f"   Average chunk size: {int(avg_length)} characters")
    print(f"   Smallest chunk: {min(chunk_lengths)} characters")
    print(f"   Largest chunk: {max(chunk_lengths)} characters")
    
    # Initialize Dense Embedding Model
    print("\n🧠 Embedding Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("   Loading model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
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
                "embedding_type": "dense"
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Create ChromaDB vector store
    print(f"\n💾 Creating Vector Database: Dense_VDBase")
    print(f"   Storage location: {persist_directory}")
    
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="Dense_VDBase",
        persist_directory=persist_directory
    )
    
    print(f"   ✅ Vector database created with {len(documents)} documents")
    
    # Test retrieval
    print("\n🔍 Testing Retrieval:")
    test_query = "What is the revenue?"
    print(f"   Query: '{test_query}'")
    
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"   Retrieved {len(results)} relevant chunks:")
    
    for i, doc in enumerate(results, 1):
        preview = doc.page_content.replace('\n', ' ')[:150]
        print(f"\n   [{i}] Chunk #{doc.metadata['chunk_id']}")
        print(f"       {preview}...")
    
    print("\n" + "=" * 70)
    print("✅ Dense Embeddings Complete!")
    print("=" * 70)
    
    return vectorstore


def main():
    # Configuration
    PDF_PATH = "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    PERSIST_DIR = "/Volumes/vibecoding/RAG-Complete Cook Book/my_vector_db/Dense_VDBase"
    
    # Create dense embeddings
    vectorstore = create_dense_embeddings(PDF_PATH, PERSIST_DIR)
    
    print("\n📋 Summary:")
    print(f"   • Embedding Type: Dense (Continuous Vectors)")
    print(f"   • Dimensions: 384")
    print(f"   • Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   • Collection: Dense_VDBase")
    print(f"   • Storage: {PERSIST_DIR}")
    print("\n✨ Dense embeddings capture semantic meaning in continuous vector space")
    print("   Best for: Semantic search, similarity matching, Q&A systems\n")


if __name__ == "__main__":
    main()
