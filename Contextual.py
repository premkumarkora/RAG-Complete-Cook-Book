"""
Contextual Embeddings with Parent-Child Indexing
Implements contextual embeddings where each chunk is enriched with surrounding context.
Uses parent-child indexing strategy: small chunks for search, large chunks for context.
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


def create_parent_child_chunks(text):
    """
    Create parent and child chunks with contextual relationships.
    
    Strategy:
    - Parent chunks: Large 1000-character blocks (provide context)
    - Child chunks: Small 200-character blocks (used for search)
    - Each child knows its parent for context retrieval
    
    Returns:
        documents: List of Document objects with parent-child metadata
    """
    # Parent splitter: Large chunks for context
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Child splitter: Small chunks for precise search
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create parent chunks
    parent_chunks = parent_splitter.split_text(text)
    
    print(f"   Parent chunks (context): {len(parent_chunks)}")
    print(f"   Average parent size: {sum(len(p) for p in parent_chunks) // len(parent_chunks)} chars")
    
    # Create child chunks from each parent
    all_documents = []
    child_counter = 0
    
    for parent_id, parent_text in enumerate(parent_chunks):
        # Split parent into children
        child_chunks = child_splitter.split_text(parent_text)
        
        # Create document for each child with parent context
        for child_text in child_chunks:
            doc = Document(
                page_content=child_text,
                metadata={
                    "chunk_id": child_counter,
                    "parent_id": parent_id,
                    "parent_text": parent_text,  # Full parent context stored
                    "child_size": len(child_text),
                    "parent_size": len(parent_text),
                    "embedding_type": "contextual"
                }
            )
            all_documents.append(doc)
            child_counter += 1
    
    print(f"   Child chunks (searchable): {len(all_documents)}")
    print(f"   Average child size: {sum(doc.metadata['child_size'] for doc in all_documents) // len(all_documents)} chars")
    
    return all_documents


def create_contextual_embeddings(pdf_path, persist_directory):
    """
    Create contextual embeddings with parent-child indexing and store in ChromaDB.
    
    Args:
        pdf_path: Path to the PDF file
        persist_directory: Directory to store the vector database
    """
    print("=" * 70)
    print("📊 CONTEXTUAL EMBEDDINGS - Parent-Child Indexing")
    print("=" * 70)
    print("\n💡 Philosophy: 'Search with a scalpel, read with a microscope'")
    print("   • Search on small, precise chunks (children)")
    print("   • Retrieve large, contextual chunks (parents)")
    
    # Extract text from PDF
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"\n📄 Processing: {pdf_file.name}")
    text = extract_text_from_pdf(pdf_path)
    print(f"   Total characters extracted: {len(text):,}")
    
    # Create parent-child chunks
    print("\n🔪 Chunking Strategy: Parent-Child Indexing")
    documents = create_parent_child_chunks(text)
    
    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = pdf_file.name
    
    # Initialize Embedding Model
    print("\n🧠 Embedding Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("   Loading model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("   ✅ Model loaded successfully")
    
    # Create ChromaDB vector store
    print(f"\n💾 Creating Vector Database: Contextual_VDBase")
    print(f"   Storage location: {persist_directory}")
    print("   Note: Child chunks are embedded, parent context stored in metadata")
    
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="Contextual_VDBase",
        persist_directory=persist_directory
    )
    
    print(f"   ✅ Vector database created with {len(documents)} child chunks")
    
    # Test retrieval with context
    print("\n🔍 Testing Contextual Retrieval:")
    test_query = "What is the revenue?"
    print(f"   Query: '{test_query}'")
    
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"   Retrieved {len(results)} relevant child chunks with parent context:")
    
    for i, doc in enumerate(results, 1):
        child_preview = doc.page_content.replace('\n', ' ')[:100]
        parent_preview = doc.metadata['parent_text'].replace('\n', ' ')[:150]
        
        print(f"\n   [{i}] Child Chunk #{doc.metadata['chunk_id']} (from Parent #{doc.metadata['parent_id']})")
        print(f"       Child: {child_preview}...")
        print(f"       Parent Context: {parent_preview}...")
    
    print("\n" + "=" * 70)
    print("✅ Contextual Embeddings Complete!")
    print("=" * 70)
    
    return vectorstore


def main():
    # Configuration
    PDF_PATH = "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    PERSIST_DIR = "/Volumes/vibecoding/RAG-Complete Cook Book/my_vector_db/Contextual_VDBase"
    
    # Create contextual embeddings
    vectorstore = create_contextual_embeddings(PDF_PATH, PERSIST_DIR)
    
    print("\n📋 Summary:")
    print(f"   • Embedding Type: Contextual (Parent-Child)")
    print(f"   • Search Units: Small child chunks (~200 chars)")
    print(f"   • Context Units: Large parent chunks (~1000 chars)")
    print(f"   • Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   • Collection: Contextual_VDBase")
    print(f"   • Storage: {PERSIST_DIR}")
    print("\n✨ Contextual embeddings provide rich context for retrieved chunks")
    print("   Best for: Q&A systems needing surrounding context, precise search with broad understanding")
    print("\n   How it works:")
    print("   1. Search matches small, precise child chunks")
    print("   2. Return includes full parent chunk for context")
    print("   3. Balances precision (child) with completeness (parent)\n")


if __name__ == "__main__":
    main()
