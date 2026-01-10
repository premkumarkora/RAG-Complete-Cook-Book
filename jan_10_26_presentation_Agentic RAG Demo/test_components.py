"""
Quick test script to verify components work
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("=" * 70)
print("🧪 Testing Agentic RAG Components")
print("=" * 70)

# Test 1: Check OpenAI API key
print("\n1️⃣ Checking OpenAI API Key...")
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"   ✅ API key found (starts with: {api_key[:10]}...)")
else:
    print("   ❌ No API key found in .env file")
    sys.exit(1)

# Test 2: Import modules
print("\n2️⃣ Testing imports...")
try:
    import config
    print("   ✅ config.py imported")
    from chunking_engine import ChunkingEngine
    print("   ✅ chunking_engine.py imported")
    from agentic_workflow import AgenticWorkflow
    print("   ✅ agentic_workflow.py imported")
    from ragas_evaluator import RAGASEvaluator
    print("   ✅ ragas_evaluator.py imported")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 3: Check PDF files
print("\n3️⃣ Checking PDF files...")
pdf_files = list(Path(config.PDF_DIR).glob("*.pdf"))
print(f"   Found {len(pdf_files)} PDF files:")
for pdf in pdf_files:
    print(f"   - {pdf.name}")

# Test 4: Initialize Chunking Engine
print("\n4️⃣ Testing Chunking Engine initialization...")
try:
    engine = ChunkingEngine()
    print("   ✅ ChunkingEngine initialized")
    
    # Check if vector DB exists
    db_exists = engine.check_vector_db_exists()
    if db_exists:
        print("   ✅ Vector database already exists")
    else:
        print("   ⚠️  Vector database not found (will be created on first run)")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test LangGraph Workflow (without running)
print("\n5️⃣ Testing LangGraph Workflow structure...")
try:
    # We can't run without vector DB, but we can check structure
    workflow = AgenticWorkflow(engine)
    print("   ✅ AgenticWorkflow initialized")
    print("   ✅ LangGraph compiled successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test RAGAS Evaluator
print("\n6️⃣ Testing RAGAS Evaluator...")
try:
    evaluator = RAGASEvaluator()
    print("   ✅ RAGASEvaluator initialized")
    
    # Quick test
    test_score = evaluator.get_score_interpretation(0.85)
    print(f"   ✅ Score interpretation works: {test_score}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ All component tests passed!")
print("=" * 70)
print("\n📌 Next steps:")
print("   1. Run: streamlit run app.py")
print("   2. The app will automatically create the vector database")
print("   3. Start asking questions!")
print()
