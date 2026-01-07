"""
Parent-Child Indexing (Small-to-Big Retrieval)
Philosophy: "Search with a scalpel, read with a microscope."

Strategy:
1. Parent Chunks: Large 1000-token blocks (stored, not embedded)
2. Child Chunks: Small 200-token blocks (embedded for search)
3. Retrieval: Search children, return parent for context
"""

from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def create_parent_child_chunks(text):
    """
    Create parent and child chunks with relationships.
    
    Returns:
        parent_docs: List of parent Document objects
        child_docs: List of child Document objects with parent_id metadata
    """
    # Create Parent Chunks (Large: ~1000 chars ≈ 250 tokens)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    parent_chunks = parent_splitter.split_text(text)
    
    # Create Child Chunks (Small: ~200 chars ≈ 50 tokens)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    parent_docs = []
    child_docs = []
    
    for parent_id, parent_text in enumerate(parent_chunks):
        # Store parent
        parent_doc = Document(
            page_content=parent_text,
            metadata={"parent_id": parent_id, "type": "parent"}
        )
        parent_docs.append(parent_doc)
        
        # Create children from this parent
        child_chunks = child_splitter.split_text(parent_text)
        for child_idx, child_text in enumerate(child_chunks):
            child_doc = Document(
                page_content=child_text,
                metadata={
                    "parent_id": parent_id,
                    "child_id": f"{parent_id}_{child_idx}",
                    "type": "child"
                }
            )
            child_docs.append(child_doc)
    
    return parent_docs, child_docs


def main():
    # PDF file
    pdf_path = "/Volumes/vibecoding/RAG-Complete Cook Book/ITC-October-Q2-2526.pdf"
    
    print("📄 Parent-Child Indexing (Small-to-Big Retrieval)")
    print('   Philosophy: "Search with a scalpel, read with a microscope."')
    print("=" * 70)
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ File not found: {pdf_file.name}")
        return
    
    print(f"\n📂 Processing: {pdf_file.name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    print(f"   Total characters: {len(text):,}")
    
    # Create parent-child structure
    print("\n📊 Creating Parent-Child Structure...")
    parent_docs, child_docs = create_parent_child_chunks(text)
    
    print(f"   ✓ Parent chunks (large, ~1000 chars): {len(parent_docs)}")
    print(f"   ✓ Child chunks (small, ~200 chars): {len(child_docs)}")
    
    # Initialize embeddings
    print("\n🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Index ONLY child chunks (for search)
    print("🔍 Indexing child chunks for search...")
    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings
    )
    
    print("✅ Vector store created with child chunks\n")
    
    # Demonstration: Search with a query
    query = "What was FMCG segment revenue growth?"
    
    print(f"{'='*70}")
    print(f"RETRIEVAL DEMONSTRATION")
    print(f"{'='*70}")
    print(f"\n🔎 Query: '{query}'")
    
    # Step 1: Search child chunks (precise)
    print("\n📌 Step 1: Searching CHILD chunks (precise, 200 chars)...")
    child_results = vectorstore.similarity_search(query, k=2)
    
    for i, child in enumerate(child_results[:2], 1):
        print(f"\n   Child Match {i}:")
        print(f"   Parent ID: {child.metadata['parent_id']}")
        print(f"   Child ID: {child.metadata['child_id']}")
        preview = child.page_content[:150].replace('\n', ' ')
        print(f"   Content: {preview}...")
    
    # Step 2: Retrieve parent chunks (context)
    print("\n\n📖 Step 2: Fetching PARENT chunks (full context, 1000 chars)...")
    
    for i, child in enumerate(child_results[:2], 1):
        parent_id = child.metadata['parent_id']
        parent_doc = parent_docs[parent_id]
        
        print(f"\n{'─'*70}")
        print(f"PARENT CHUNK {i} (ID: {parent_id})")
        print(f"{'─'*70}")
        print(f"Length: {len(parent_doc.page_content)} chars")
        print(f"\nFull Content:")
        # Show first 400 chars of parent
        preview = parent_doc.page_content[:400].replace('\n', ' ')
        print(f"{preview}...")
    
    print(f"\n{'='*70}")
    print("\n✨ Why This Works:")
    print("   ✅ Search: Small child chunks = precise matching")
    print("   ✅ Context: Large parent chunks = full context for LLM")
    print("   ✅ Best of both worlds: Precision + Context")
    
    # Show statistics
    print("\n📊 Size Comparison:")
    avg_child = sum(len(c.page_content) for c in child_docs) / len(child_docs)
    avg_parent = sum(len(p.page_content) for p in parent_docs) / len(parent_docs)
    print(f"   Average child size: {int(avg_child)} chars (~{int(avg_child/4)} tokens)")
    print(f"   Average parent size: {int(avg_parent)} chars (~{int(avg_parent/4)} tokens)")
    print(f"   Context gain: {int(avg_parent/avg_child)}x more context in parent")


if __name__ == "__main__":
    main()
"""
📄 Parent-Child Indexing (Small-to-Big Retrieval)
   Philosophy: "Search with a scalpel, read with a microscope."
======================================================================

📂 Processing: ITC-October-Q2-2526.pdf
   Total characters: 18,430

📊 Creating Parent-Child Structure...
   ✓ Parent chunks (large, ~1000 chars): 21
   ✓ Child chunks (small, ~200 chars): 131

🧠 Loading embedding model...
🔍 Indexing child chunks for search...
✅ Vector store created with child chunks

======================================================================
RETRIEVAL DEMONSTRATION
======================================================================

🔎 Query: 'What was FMCG segment revenue growth?'

📌 Step 1: Searching CHILD chunks (precise, 200 chars)...

   Child Match 1:
   Parent ID: 0
   Child ID: 0_6
   Content: • Gross Revenue (ex-Agri Business) up 7.9% YoY; EBITDA up 2.2% YoY.    • FMCG – Others Segment sustained its Revenue growth momentum amidst operationa...

   Child Match 2:
   Parent ID: 6
   Child ID: 6_3
   Content: FMCG – OTHERS    • The FMCG Businesses sustained its Revenue growth momentum amidst operational challenges; up  8% YoY ex-Notebooks...


📖 Step 2: Fetching PARENT chunks (full context, 1000 chars)...

──────────────────────────────────────────────────────────────────────
PARENT CHUNK 1 (ID: 0)
──────────────────────────────────────────────────────────────────────
Length: 975 chars

Full Content:
1  FMCG ⚫ PAPERBOARDS & PACKAGING ⚫ AGRI-BUSINESS  ⚫ INFORMATION TECHNOLOGY  Visit us at www.itcportal.com ⚫ Corporate Identity Number : L16005WB1910PLC001985 ⚫ e-mail : enduringvalue@itc.in      Media Statement  October 30, 2025  Financial Results for the Quarter ended 30th September, 2025  Highlights  Standalone  • Gross Revenue at Rs. 19148 cr.; up 7.1% YoY (ex-Agri Business) driven by Cigarett...

──────────────────────────────────────────────────────────────────────
PARENT CHUNK 2 (ID: 6)
──────────────────────────────────────────────────────────────────────
Length: 956 chars

Full Content:
especially for the FMCG categories, causing short -term business disruptions during the quarter.  Notwithstanding such transitory factors, the Company delivered resilient performance during the quarter.  Gross Revenue2 stood at Rs. 19,148 crores, while PBT and PAT stood at Rs. 6,851 crores and Rs. 5180 crores  respectively. Earnings Per Share for the quarter stood at Rs. 4.13 (LY 3.98).    FMCG – ...

======================================================================

✨ Why This Works:
   ✅ Search: Small child chunks = precise matching
   ✅ Context: Large parent chunks = full context for LLM
   ✅ Best of both worlds: Precision + Context

📊 Size Comparison:
   Average child size: 148 chars (~37 tokens)
   Average parent size: 934 chars (~233 tokens)
   Context gain: 6x more context in parent

"""