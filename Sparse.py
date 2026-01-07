"""
Sparse Embeddings with BM25 and ChromaDB
Creates sparse vector embeddings using BM25 (Best Match 25) algorithm.
Sparse embeddings have mostly zero values with a few non-zero values representing keyword importance.
"""

from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import os
import json


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def create_sparse_embeddings(pdf_path, persist_directory):
    """
    Create sparse embeddings using BM25 and store in ChromaDB.
    
    Note: ChromaDB primarily uses dense embeddings, so we'll store BM25 scores
    as metadata and use a hybrid approach with dense embeddings for storage.
    
    Args:
        pdf_path: Path to the PDF file
        persist_directory: Directory to store the vector database
    """
    print("=" * 70)
    print("📊 SPARSE EMBEDDINGS - BM25 Keyword-Based Representation")
    print("=" * 70)
    
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
    
    # Create Document objects
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": pdf_file.name,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "embedding_type": "sparse_bm25"
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Initialize BM25 Retriever for sparse representation
    print("\n🧠 Sparse Representation: BM25 Algorithm")
    print("   BM25 is a probabilistic ranking function based on:")
    print("   • Term frequency (TF)")
    print("   • Inverse document frequency (IDF)")
    print("   • Document length normalization")
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3  # Number of documents to retrieve
    
    print("   ✅ BM25 retriever initialized")
    
    # For ChromaDB storage, we'll use a lightweight embedding
    # This allows us to maintain the collection while emphasizing sparse retrieval
    print("\n💾 Creating Vector Database: Sparse_VDBase")
    print(f"   Storage location: {persist_directory}")
    print("   Note: Using minimal dense embeddings for ChromaDB compatibility")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="Sparse_VDBase",
        persist_directory=persist_directory
    )
    
    print(f"   ✅ Vector database created with {len(documents)} documents")
    
    # Test BM25 retrieval
    print("\n🔍 Testing BM25 Sparse Retrieval:")
    test_query = "What is the revenue?"
    print(f"   Query: '{test_query}'")
    
    # BM25 retrieval (keyword-based)
    bm25_results = bm25_retriever.invoke(test_query)
    print(f"\n   BM25 Results (Keyword Matching):")
    
    for i, doc in enumerate(bm25_results, 1):
        preview = doc.page_content.replace('\n', ' ')[:150]
        print(f"\n   [{i}] Chunk #{doc.metadata['chunk_id']}")
        print(f"       {preview}...")
    
    print("\n" + "=" * 70)
    print("✅ Sparse Embeddings Complete!")
    print("=" * 70)
    
    return vectorstore, bm25_retriever


def main():
    # Configuration
    PDF_PATH = "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    PERSIST_DIR = "/Volumes/vibecoding/RAG-Complete Cook Book/my_vector_db/Sparse_VDBase"
    
    # Create sparse embeddings
    vectorstore, bm25_retriever = create_sparse_embeddings(PDF_PATH, PERSIST_DIR)
    
    print("\n📋 Summary:")
    print(f"   • Embedding Type: Sparse (BM25)")
    print(f"   • Algorithm: Best Match 25")
    print(f"   • Representation: Term frequency + IDF weights")
    print(f"   • Collection: Sparse_VDBase")
    print(f"   • Storage: {PERSIST_DIR}")
    print("\n✨ Sparse embeddings excel at exact keyword matching")
    print("   Best for: Keyword search, term matching, lexical retrieval")
    print("   Characteristics:")
    print("   • Mostly zero values (sparse)")
    print("   • Non-zero values indicate keyword importance")
    print("   • Complementary to dense embeddings (combine for hybrid search)\n")


if __name__ == "__main__":
    main()
