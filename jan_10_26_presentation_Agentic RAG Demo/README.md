# Agentic RAG Demo

A comprehensive RAG (Retrieval-Augmented Generation) demonstration featuring advanced chunking strategies, LangGraph-based agentic workflow, and automated quality evaluation using RAGAS.

## 🌟 Features

### 1. **Smart Initialization**
- Auto-detects existing vector database
- Automatically processes PDFs if database is missing
- Live visualization of chunking process

### 2. **Dual Chunking Strategy**
- **Contextual Chunking**: Adds document context to each chunk for better semantic understanding
- **Propositional Chunking**: Uses LLM to break complex sentences into standalone, searchable facts

### 3. **LangGraph Agentic Workflow**
Three-node StateGraph implementation:
- **Retrieve Node**: Fetches relevant documents from ChromaDB
- **Grade Node**: LLM evaluates document relevance (prevents hallucinations)
- **Generate Node**: Creates answers using only relevant documents

### 4. **RAGAS Evaluation Framework**
Automated quality assessment with three metrics:
- **Faithfulness**: Detects hallucinations (answer must come from context)
- **Answer Relevance**: Checks if answer addresses the question
- **Context Precision**: Evaluates retrieval quality

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd "/Volumes/vibecoding/RAG-Complete Cook Book/jan_10_26_presentation_Agentic RAG Demo"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

4. **Run the application:**
```bash
streamlit run app.py
```

The app will automatically:
- Check for existing vector database in `vector_db/` directory
- If not found, process the PDFs (`ITC-August-Q1-2526.pdf` and `ITC-October-Q2-2526.pdf`)
- Show live chunking visualization
- Create and persist the vector database

## 📖 Usage

### First Launch
On first launch, the app will:
1. Display the chunking process in real-time
2. Show examples of contextual chunks (with document context)
3. Show examples of propositions (simple facts extracted by LLM)
4. Create a ChromaDB vector database with all chunks

### Asking Questions
1. Once the vector database is ready, enter your question in the input field
2. Click "Search & Answer"
3. View three tabs:
   - **LangGraph Workflow**: See the step-by-step execution
   - **Answer**: View the final answer and source documents
   - **RAGAS Evaluation**: See quality metrics

### Example Questions
- "What was ITC's gross revenue growth in Q1 2025?"
- "Tell me about ITC's Agri Business performance"
- "What is ITCMAARS?"
- "What are ITC's sustainability achievements?"

## 🏗️ Architecture

### Project Structure
```
jan_10_26_presentation_Agentic RAG Demo/
├── app.py                      # Main Streamlit application
├── config.py                   # Centralized configuration
├── chunking_engine.py          # Dual chunking implementation
├── agentic_workflow.py         # LangGraph StateGraph
├── ragas_evaluator.py          # RAGAS evaluation metrics
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── ITC-August-Q1-2526.pdf     # Sample PDF 1
├── ITC-October-Q2-2526.pdf    # Sample PDF 2
└── vector_db/                 # ChromaDB storage (auto-created)
```

### Data Flow

```
PDF Documents
    ↓
Dual Chunking (Contextual + Propositional)
    ↓
Vector Embeddings (OpenAI)
    ↓
ChromaDB Vector Database
    ↓
User Question → LangGraph Workflow:
    1. Retrieve (fetch documents)
    2. Grade (check relevance)
    3. Generate (create answer)
    ↓
Answer + RAGAS Evaluation
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model Settings
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking Parameters
CONTEXTUAL_CHUNK_SIZE = 1000
CONTEXTUAL_CHUNK_OVERLAP = 200

# Retrieval Parameters
TOP_K_RETRIEVE = 5
RELEVANCE_THRESHOLD = 0.6
```

## 📊 Understanding RAGAS Metrics

### Faithfulness (0.0 - 1.0)
- **What**: Checks if the answer uses ONLY information from retrieved documents
- **High Score**: Answer is faithful to context
- **Low Score**: Answer contains hallucinated information

### Answer Relevance (0.0 - 1.0)
- **What**: Evaluates if the answer addresses the user's question
- **High Score**: Answer directly addresses the question
- **Low Score**: Answer is off-topic or doesn't answer the question

### Context Precision (0.0 - 1.0)
- **What**: Measures retrieval quality
- **High Score**: Retrieved documents are relevant
- **Low Score**: Retrieved documents aren't relevant to the question

## 🎯 Key Capabilities

### Chunking Strategies

**Contextual Chunking:**
- Adds document name and position metadata
- Helps maintain semantic context
- Better for complex documents

**Propositional Chunking:**
- LLM breaks complex sentences into simple facts
- Each proposition is independently searchable
- Ideal for high-precision retrieval

### LangGraph Workflow

The **Grade node** is crucial:
- Prevents hallucinations by filtering irrelevant documents
- Only relevant documents proceed to generation
- If no relevant documents found, returns "No Data" response

### Live Visualization

- **Chunking Process**: See chunks being created in real-time
- **Workflow Execution**: Track each LangGraph node execution
- **RAGAS Metrics**: Immediate quality feedback

## 🔄 Resetting the Database

To recreate the vector database:
1. Use the "Reset Vector Database" button in the sidebar
2. Or manually delete the `vector_db/` directory
3. Restart the app

## 🛠️ Troubleshooting

**Issue**: "No vector database found" on every launch
- **Solution**: Ensure `vector_db/` directory has write permissions

**Issue**: OpenAI API errors
- **Solution**: Check your API key in `.env` file and ensure you have credits

**Issue**: Slow chunking process
- **Solution**: Normal for first run. Propositional chunking uses LLM calls which take time.

## 📝 Notes

- The app uses `gpt-4o-mini` for cost-effectiveness. You can switch to `gpt-4` in `config.py` for better quality.
- Propositional chunking processes only the first 10 paragraphs per document for demo purposes.
- Vector database is persistent and only needs to be created once.

## 🤝 Contributing

This is a demonstration project. Feel free to extend it with:
- More chunking strategies
- Additional RAGAS metrics
- Query rewriting in the workflow
- Multiple vector database support

## 📄 License

Educational demonstration project.

---

**Built with**: Streamlit • LangGraph • LangChain • ChromaDB • OpenAI • RAGAS
