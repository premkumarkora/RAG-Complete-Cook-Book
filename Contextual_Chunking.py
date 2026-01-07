"""
Contextual Retrieval (2025/2026 Standard)
Solves the "Lost in the Middle" problem by prepending context to each chunk
"""

import os
from pypdf import PdfReader
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def create_base_chunks(text, chunk_size=800, chunk_overlap=100):
    """Create initial chunks using recursive splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)


def generate_context(chunk, document_name, llm):
    """
    Use LLM to generate contextual information for a chunk.
    
    This prepends context like:
    [Context: From ITC Q1 2025 Financial Report, Revenue Section]
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a document analyzer. Generate a concise 1-sentence context description for the given text chunk."),
        ("user", """Document: {doc_name}

Chunk:
{chunk}

Generate a brief context sentence (max 20 words) that describes what this chunk is about and where it's from. Format: "Context: [your description]"

Just return the context line, nothing else.""")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "doc_name": document_name,
        "chunk": chunk[:500]  # Only send first 500 chars to save tokens
    })
    
    return response.content.strip()


def main():
    # Load environment variables
    load_dotenv()
    
    # PDF files
    pdfs = [
        "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-August-Q1-2526.pdf",
        "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    ]
    
    print("📄 Contextual Retrieval - The 2025/2026 Standard")
    print("   Solves: 'Lost in the Middle' problem")
    print("   Method: LLM prepends context to each chunk")
    print("=" * 70)
    
    # Initialize LLM (using fast, cheap model)
    print("\n🤖 Initializing LLM (gpt-5-nano for context generation)...")
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    print("✅ LLM ready. Processing PDFs...\n")
    
    # Process only first PDF for demo (to save time/tokens)
    pdf_path = pdfs[0]
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_file.name}")
        return
    
    print(f"📂 Processing: {pdf_file.name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    # Create base chunks
    print("   Creating base chunks...")
    base_chunks = create_base_chunks(text, chunk_size=800, chunk_overlap=100)
    
    print(f"   Total chunks: {len(base_chunks)}")
    
    # Generate contextualized versions (only for first 3 chunks to save time)
    print("\n   Generating contextualized versions (first 3 chunks)...")
    
    for i in range(min(3, len(base_chunks))):
        chunk = base_chunks[i]
        
        print(f"\n{'='*70}")
        print(f"CHUNK {i+1} COMPARISON")
        print(f"{'='*70}")
        
        # Show original chunk
        print("\n📋 ORIGINAL CHUNK:")
        preview = chunk[:300].replace('\n', ' ')
        print(f"   {preview}...")
        print(f"   Length: {len(chunk)} chars")
        
        # Generate and show contextualized version
        print("\n🎯 CONTEXTUALIZED VERSION:")
        context = generate_context(chunk, pdf_file.stem, llm)
        contextualized_chunk = f"{context}\n\n{chunk}"
        
        context_preview = contextualized_chunk[:400].replace('\n', ' ')
        print(f"   {context_preview}...")
        print(f"   Length: {len(contextualized_chunk)} chars")
        
        print(f"\n💡 Why this helps:")
        print(f"   Even if search query doesn't match exact words in chunk,")
        print(f"   the context provides searchable keywords and metadata.")
    
    print("\n" + "=" * 70)
    print("\n✨ Key Benefit:")
    print("   Chunks now carry their own context, making them findable")
    print("   even when key terms are in different paragraphs.")


if __name__ == "__main__":
    main()
