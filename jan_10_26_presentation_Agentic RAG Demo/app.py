"""
Agentic RAG Demo - Streamlit Application
Features: Dual Chunking, LangGraph Workflow, RAGAS Evaluation
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Import our modules
import config
from chunking_engine import ChunkingEngine
from agentic_workflow import AgenticWorkflow
from ragas_evaluator import RAGASEvaluator

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chunk-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .workflow-step {
        background-color: #e8f4f8;
        border-left: 4px solid #2ca02c;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 4px;
        font-family: monospace;
        color: #000000;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .score-excellent { color: #00C853; font-weight: bold; }
    .score-good { color: #2196F3; font-weight: bold; }
    .score-fair { color: #FF9800; font-weight: bold; }
    .score-poor { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.chunking_engine = None
        st.session_state.workflow = None
        st.session_state.evaluator = None
        st.session_state.vector_db_ready = False


def display_header():
    """Display application header"""
    st.markdown('<div class="main-header">🤖 Agentic RAG Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Contextual + Propositional Chunking • LangGraph Workflow • RAGAS Evaluation</div>',
        unsafe_allow_html=True
    )


def check_and_initialize_vectordb():
    """Check if vector DB exists, create if not"""
    st.header("📊 Vector Database Initialization")
    
    # Initialize chunking engine
    if not st.session_state.chunking_engine:
        st.session_state.chunking_engine = ChunkingEngine()
    
    chunking_engine = st.session_state.chunking_engine
    
    # Check if vector DB exists
    if chunking_engine.check_vector_db_exists():
        st.success("✅ Vector database found! Ready to answer questions.")
        st.session_state.vector_db_ready = True
        return True
    
    # Need to create vector DB
    st.warning("⚠️ No vector database found. Creating from PDFs...")
    
    # Find PDF files
    pdf_dir = Path(config.PDF_DIR)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        st.error("❌ No PDF files found in directory!")
        return False
    
    st.info(f"📄 Found {len(pdf_files)} PDF files: {', '.join([f.name for f in pdf_files])}")
    
    # Chunking visualization area
    st.subheader("🔍 Live Chunking Process")
    
    chunk_display = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process PDFs
    chunks_shown = {"contextual": 0, "propositional": 0}
    
    for chunk_data in chunking_engine.process_pdfs([str(f) for f in pdf_files]):
        
        if chunk_data.get("status") == "processing_document":
            status_text.markdown(f"**{chunk_data['message']}**")
        
        elif chunk_data.get("status") == "chunking":
            status_text.markdown(f"**{chunk_data['message']}**")
        
        elif chunk_data.get("type") == "contextual":
            if chunks_shown["contextual"] < 3:  # Show first 3
                with chunk_display:
                    with st.expander(f"📝 Contextual Chunk {chunk_data['chunk_id']}", expanded=False):
                        st.text(chunk_data["content"][:500] + "...")
                        st.caption(f"Metadata: {chunk_data['metadata']}")
                chunks_shown["contextual"] += 1
        
        elif chunk_data.get("type") == "propositional":
            if chunks_shown["propositional"] < 5:  # Show first 5
                with chunk_display:
                    st.markdown(f'<div class="chunk-box">🎯 <strong>Proposition:</strong> {chunk_data["content"]}</div>', 
                              unsafe_allow_html=True)
                chunks_shown["propositional"] += 1
        
        elif chunk_data.get("status") == "vectorizing":
            status_text.markdown(f"**{chunk_data['message']}**")
            progress_bar.progress(0.9)
        
        elif chunk_data.get("status") == "complete":
            progress_bar.progress(1.0)
            status_text.markdown(f"**{chunk_data['message']}**")
            st.success(f"✅ Created {chunk_data['total_chunks']} total chunks!")
            st.session_state.vector_db_ready = True
            time.sleep(1)
            return True
    
    return False


def display_question_interface():
    """Display question input and workflow"""
    st.header("❓ Ask Questions")
    
    # Initialize workflow if needed
    if not st.session_state.workflow:
        st.session_state.workflow = AgenticWorkflow(st.session_state.chunking_engine)
    
    if not st.session_state.evaluator:
        st.session_state.evaluator = RAGASEvaluator()
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What was ITC's gross revenue growth in Q1 2025?",
        key="question_input"
    )
    
    if st.button("🔍 Search & Answer", type="primary"):
        if question:
            process_question(question)
        else:
            st.warning("Please enter a question first!")


def process_question(question: str):
    """Process question through workflow and display results"""
    
    # Create tabs for workflow and results
    workflow_tab, results_tab, ragas_tab = st.tabs([
        "🔄 LangGraph Workflow", 
        "💬 Answer", 
        "📊 RAGAS Evaluation"
    ])
    
    with workflow_tab:
        st.subheader("LangGraph Workflow Execution")
        workflow_container = st.container()
        
        with st.spinner("Running agentic workflow..."):
            # Run workflow
            final_state = st.session_state.workflow.run(question)
            
            # Display workflow history
            with workflow_container:
                for step in final_state["workflow_history"]:
                    st.markdown(f'<div class="workflow-step">{step}</div>', unsafe_allow_html=True)
                    time.sleep(0.1)  # Small delay for visualization
    
    with results_tab:
        st.subheader("Final Answer")
        
        # Display answer
        answer = final_state["final_answer"]
        st.markdown(f"**Answer:**")
        st.info(answer)
        
        # Display retrieved documents
        if final_state.get("graded_docs"):
            with st.expander(f"📚 View {len(final_state['graded_docs'])} Relevant Documents"):
                for i, doc in enumerate(final_state["graded_docs"], 1):
                    st.markdown(f"**Document {i}:**")
                    st.text(doc["content"][:300] + "...")
                    st.caption(f"Metadata: {doc.get('metadata', {})}")
                    st.divider()
    
    with ragas_tab:
        st.subheader("RAGAS Quality Metrics")
        
        with st.spinner("Evaluating with RAGAS framework..."):
            # Evaluate
            scores = st.session_state.evaluator.evaluate_all(
                question=question,
                answer=final_state["final_answer"],
                contexts=final_state.get("graded_docs", []),
                relevance_scores=final_state.get("relevance_scores", [])
            )
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score = scores["faithfulness"]
                interpretation, _ = st.session_state.evaluator.get_score_interpretation(score)
                st.metric(
                    label="Faithfulness",
                    value=f"{score:.2f}",
                    help="Did the answer come only from the context? (Detects hallucinations)"
                )
                st.caption(interpretation)
            
            with col2:
                score = scores["answer_relevance"]
                interpretation, _ = st.session_state.evaluator.get_score_interpretation(score)
                st.metric(
                    label="Answer Relevance",
                    value=f"{score:.2f}",
                    help="Did the answer address the question?"
                )
                st.caption(interpretation)
            
            with col3:
                score = scores["context_precision"]
                interpretation, _ = st.session_state.evaluator.get_score_interpretation(score)
                st.metric(
                    label="Context Precision",
                    value=f"{score:.2f}",
                    help="Did retrieval find the right documents?"
                )
                st.caption(interpretation)
            
            # Overall score
            st.divider()
            overall_score = scores["average"]
            overall_interpretation, _ = st.session_state.evaluator.get_score_interpretation(overall_score)
            
            st.markdown(f"### Overall Quality: {overall_score:.2f} - {overall_interpretation}")
            
            # Explanation
            with st.expander("📖 Understanding RAGAS Metrics"):
                st.markdown("""
                **Faithfulness (Hallucination Detection):**
                - Checks if the answer uses ONLY information from retrieved documents
                - Low score = Answer contains hallucinated information
                
                **Answer Relevance:**
                - Evaluates if the answer actually addresses the user's question
                - Low score = Answer is off-topic or doesn't answer the question
                
                **Context Precision:**
                - Measures retrieval quality (did we fetch the right documents?)
                - Low score = Retrieved documents weren't relevant
                """)


def display_sidebar():
    """Display sidebar with info and configuration"""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This demo showcases an advanced **Agentic RAG** system with:
        
        **🔍 Dual Chunking:**
        - Contextual chunking (adds document context)
        - Propositional chunking (LLM extracts facts)
        
        **🤖 LangGraph Workflow:**
        - **Retrieve**: Fetch documents from ChromaDB
        - **Grade**: LLM checks relevance (prevents hallucinations)
        - **Generate**: Create answer from graded docs
        
        **📊 RAGAS Evaluation:**
        - Faithfulness (hallucination detection)
        - Answer Relevance
        - Context Precision
        """)
        
        st.divider()
        
        st.header("⚙️ Configuration")
        st.code(f"""
Model: {config.MODEL_NAME}
Chunk Size: {config.CONTEXTUAL_CHUNK_SIZE}
Top-K Retrieval: {config.TOP_K_RETRIEVE}
        """)
        
        st.divider()
        
        if st.button("🔄 Reset Vector Database"):
            import shutil
            if config.VECTOR_DB_PATH.exists():
                shutil.rmtree(config.VECTOR_DB_PATH)
                st.session_state.vector_db_ready = False
                st.session_state.chunking_engine = None
                st.success("Vector database deleted! Refresh to recreate.")
                st.rerun()


def main():
    """Main application"""
    initialize_session_state()
    display_header()
    display_sidebar()
    
    # Check and initialize vector database
    if not st.session_state.vector_db_ready:
        if check_and_initialize_vectordb():
            st.rerun()
    else:
        # Vector DB ready, show question interface
        display_question_interface()


if __name__ == "__main__":
    main()
