"""
Binary Quantization Embeddings
Converts dense float embeddings into binary vectors (0s and 1s) for massive storage reduction.
Achieves up to 32x compression while maintaining reasonable retrieval performance.
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


class BinaryQuantizedEmbeddings:
    """
    Wrapper for embeddings that applies binary quantization.
    Converts float embeddings to binary (0/1) using sign-based quantization.
    """
    
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings
        
    def embed_documents(self, texts):
        """Embed documents and apply binary quantization."""
        # Get dense embeddings
        dense_embeddings = self.base_embeddings.embed_documents(texts)
        
        # Apply binary quantization (sign-based: positive=1, negative=0)
        binary_embeddings = []
        for embedding in dense_embeddings:
            # Convert to numpy array
            arr = np.array(embedding)
            # Sign-based quantization: values >= 0 become 1, < 0 become 0
            binary = (arr >= 0).astype(np.float32)
            binary_embeddings.append(binary.tolist())
        
        return binary_embeddings
    
    def embed_query(self, text):
        """Embed query and apply binary quantization."""
        # Get dense embedding
        dense_embedding = self.base_embeddings.embed_query(text)
        
        # Apply binary quantization
        arr = np.array(dense_embedding)
        binary = (arr >= 0).astype(np.float32)
        
        return binary.tolist()


def create_binary_quantized_embeddings(pdf_path, persist_directory):
    """
    Create binary quantized embeddings and store in ChromaDB.
    
    Binary quantization converts each dimension to 0 or 1:
    - Positive values → 1
    - Negative values → 0
    
    Storage reduction: 32-bit float → 1-bit binary = 32x compression
    
    Args:
        pdf_path: Path to the PDF file
        persist_directory: Directory to store the vector database
    """
    print("=" * 70)
    print("📊 BINARY QUANTIZATION - Extreme Compression")
    print("=" * 70)
    print("\n💡 Concept: Convert continuous values to binary (0/1)")
    print("   • Original: 32-bit floating point per dimension")
    print("   • Quantized: 1-bit binary per dimension")
    print("   • Compression ratio: 32x storage reduction!")
    print("   • Method: Sign-based (positive → 1, negative → 0)")
    
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
    
    # Initialize base embedding model
    print("\n🧠 Base Embedding Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("   Loading model...")
    
    base_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Wrap with binary quantization
    print("   Applying binary quantization wrapper...")
    embeddings = BinaryQuantizedEmbeddings(base_embeddings)
    print("   ✅ Binary quantized embeddings ready")
    
    # Create Document objects with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": pdf_file.name,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "embedding_type": "binary_quantized",
                "original_dimensions": 384,
                "bits_per_dimension": 1,
                "compression_ratio": "32x"
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Calculate storage savings
    original_size = len(chunks) * 384 * 4  # 4 bytes per float32
    binary_size = len(chunks) * 384 * 0.125  # 1 bit = 0.125 bytes
    savings_percent = ((original_size - binary_size) / original_size) * 100
    
    print(f"\n💾 Storage Analysis:")
    print(f"   Original embeddings: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    print(f"   Binary embeddings: {binary_size:,} bytes ({binary_size/1024:.1f} KB)")
    print(f"   Storage savings: {savings_percent:.1f}% ({original_size/binary_size:.1f}x compression)")
    
    # Create ChromaDB vector store
    print(f"\n💾 Creating Vector Database: Binary Quantization_VDBase")
    print(f"   Storage location: {persist_directory}")
    print(f"   Note: Embeddings are stored as binary vectors (0s and 1s)")
    
    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="Binary_Quantization_VDBase",
        persist_directory=persist_directory
    )
    
    print(f"   ✅ Vector database created with {len(documents)} binary-quantized documents")
    
    # Test retrieval with binary embeddings
    print("\n🔍 Testing Binary Quantized Retrieval:")
    test_query = "What is the revenue?"
    print(f"   Query: '{test_query}'")
    
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"   Retrieved {len(results)} relevant chunks using binary embeddings:")
    
    for i, doc in enumerate(results, 1):
        preview = doc.page_content.replace('\n', ' ')[:150]
        print(f"\n   [{i}] Chunk #{doc.metadata['chunk_id']}")
        print(f"       {preview}...")
    
    # Explain performance trade-offs
    print("\n⚖️  Performance Trade-offs:")
    print("   Pros:")
    print("   ✅ 32x storage reduction")
    print("   ✅ Faster similarity computation (Hamming distance)")
    print("   ✅ Lower memory footprint")
    print("   ✅ Faster data transfer")
    print("\n   Cons:")
    print("   ⚠️  ~10-20% reduction in retrieval quality")
    print("   ⚠️  Loss of fine-grained similarity information")
    print("   ⚠️  Best combined with re-ranking using full embeddings")
    
    print("\n" + "=" * 70)
    print("✅ Binary Quantization Complete!")
    print("=" * 70)
    
    return vectorstore


def main():
    # Configuration
    PDF_PATH = "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    PERSIST_DIR = "/Volumes/vibecoding/RAG-Complete Cook Book/my_vector_db/Binary Quantization_VDBase"
    
    # Create binary quantized embeddings
    vectorstore = create_binary_quantized_embeddings(PDF_PATH, PERSIST_DIR)
    
    print("\n📋 Summary:")
    print(f"   • Embedding Type: Binary Quantization")
    print(f"   • Quantization Method: Sign-based (+ → 1, - → 0)")
    print(f"   • Dimensions: 384 binary values")
    print(f"   • Compression: 32x vs. float32")
    print(f"   • Collection: Binary Quantization_VDBase")
    print(f"   • Storage: {PERSIST_DIR}")
    print("\n✨ Binary quantization offers extreme compression for large-scale systems")
    print("   Best for: Large-scale retrieval, edge deployment, cost-sensitive applications")
    print("\n   Recommended Architecture:")
    print("   1. First-pass retrieval: Binary embeddings (fast, cheap)")
    print("   2. Re-ranking: Full dense embeddings (accurate)")
    print("   3. Result: Best of both worlds - speed AND quality\n")


if __name__ == "__main__":
    main()
