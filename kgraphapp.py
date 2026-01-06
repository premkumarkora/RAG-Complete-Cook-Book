import streamlit as st
import os
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain / Graph Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from streamlit_agraph import agraph, Node, Edge, Config
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage

# --- 1. CONFIGURATION & STATE INIT ---
st.set_page_config(layout="wide", page_title="Agentic GraphRAG Explorer")
load_dotenv()

# Verify API Keys
if not os.getenv("NEO4J_PASSWORD") or not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Missing .env keys")
    st.stop()

# Initialize State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trace_logs" not in st.session_state:
    st.session_state.trace_logs = [] # Stores structured log objects
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

# --- 2. ADVANCED CALLBACK HANDLER (XAI) ---
class ExplainableTraceCallback(BaseCallbackHandler):
    """Captures detailed logs, inputs, and outputs for Explainable AI (XAI)"""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Log the System/User prompt (The "Thought" Process)
        st.session_state.trace_logs.append({
            "type": "thought",
            "content": "🤔 **Agent is Thinking...** (Planning next step)",
            "details": prompts[0] if prompts else "No prompt data"
        })

    def on_tool_start(self, serialized, input_str, **kwargs):
        # Log the Tool Call with JSON Input
        tool_name = serialized.get("name")
        try:
            input_json = json.loads(input_str)
        except:
            input_json = input_str # Fallback if not valid JSON
            
        st.session_state.trace_logs.append({
            "type": "tool_call",
            "tool": tool_name,
            "input": input_json,
            "message": f"🔧 **Agent Calling Tool:** `{tool_name}`"
        })

    def on_tool_end(self, output, **kwargs):
        # Log the Tool Output (Observation)
        st.session_state.trace_logs.append({
            "type": "tool_result",
            "output": output,
            "message": "✅ **Tool Returned Results**"
        })
        
    def on_agent_action(self, action, **kwargs):
        # Log the specific action decision
        st.session_state.trace_logs.append({
            "type": "action",
            "content": f"⚡ **Decided Action:** {action.tool}",
            "log": action.log
        })

# --- 3. BACKEND SETUP ---
@st.cache_resource
def get_llm_and_graph():
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    return graph, llm

graph, llm = get_llm_and_graph()

# --- 4. SIDEBAR: DATA IMPORT ---
with st.sidebar:
    st.header("📂 Data Import")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and st.button("Process & Ingest"):
        with st.spinner("Processing PDF..."):
            try:
                # 1. Read PDF
                pdf_reader = PdfReader(uploaded_file)
                text = "".join([page.extract_text() for page in pdf_reader.pages])
                
                # 2. Chunk
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
                
                # 3. Graph Extraction
                st.write("🕵️ Extracting Entities...")
                llm_transformer = LLMGraphTransformer(llm=llm)
                graph_documents = llm_transformer.convert_to_graph_documents(docs)
                
                graph.add_graph_documents(graph_documents)
                graph.refresh_schema()
                
                # 4. Vector Index
                st.write("🔤 Creating Vector Index...")
                st.session_state.vector_index = Neo4jVector.from_documents(
                    docs, OpenAIEmbeddings(),
                    url=os.getenv("NEO4J_URI"),
                    username=os.getenv("NEO4J_USERNAME"),
                    password=os.getenv("NEO4J_PASSWORD"),
                    index_name="rag_app_index"
                )
                st.success(f"✅ Ingestion Complete! Extracted {len(graph_documents[0].nodes)} nodes.")
            except Exception as e:
                st.error(f"Ingestion Error: {e}")

# --- 5. TOOL DEFINITIONS ---
@tool
def lookup_knowledge_graph(question: str) -> str:
    """Useful for finding specific entities, relationships, roles, and ownerships. Use this for questions like 'Who is X?', 'What did Y do?'."""
    try:
        # We use a separate chain so we can keep the main agent loop clean
        chain = GraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True, verbose=True)
        return chain.invoke({"query": question})['result']
    except Exception as e:
        return f"Graph Error: {e}"

@tool
def search_documents(query: str) -> str:
    """Useful for finding context, specific phrases, news, and details from the uploaded documents."""
    idx = st.session_state.get("vector_index")
    if idx:
        try:
            results = idx.similarity_search(query, k=2)
            if results:
                return "\n".join([r.page_content for r in results])
        except Exception as e:
            return f"Vector Search Error: {e}"
    return "No relevant documents found."

tools = [lookup_knowledge_graph, search_documents]
agent_executor = create_react_agent(llm, tools)

# --- 6. SYSTEM PROMPT (The "Context" Fix) ---
SYSTEM_PROMPT = """You are an intelligent analyst. 
You have access to a Knowledge Graph (structured data) and a Document Store (unstructured text).
Your job is to answer the user's questions based ONLY on the provided context.
If the documents mention specific people (like 'PremKumar'), you MUST answer about them.
Do not refuse to answer questions about people found in the documents.
Always check the Knowledge Graph first for relationships, then the Documents for details."""

# --- 7. MAIN UI LAYOUT ---
st.title("🤖 Agentic GraphRAG Explorer")
st.markdown("This interface provides a **Live Execution Trace** showing the Agent's internal reasoning and JSON data exchange.")

col1, col2 = st.columns([1, 1])

# LEFT COLUMN: CHAT
with col1:
    st.subheader("💬 Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about PremKumar or Project Purple...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Clear logs for new run
        st.session_state.trace_logs = []

        with st.chat_message("assistant"):
            with st.spinner("Agent is reasoning..."):
                try:
                    # Initialize Callback
                    trace_callback = ExplainableTraceCallback()
                    
                    # Inject System Prompt safely into messages list
                    messages = [SystemMessage(content=SYSTEM_PROMPT)] + [("user", user_input)]
                    
                    response = agent_executor.invoke(
                        {"messages": messages},
                        config={"callbacks": [trace_callback]} 
                    )
                    
                    final_answer = response["messages"][-1].content
                    st.write(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                except Exception as e:
                    st.error(f"Agent Error: {e}")

# RIGHT COLUMN: VISUALIZATION (THE NEW LOGIC)
with col2:
    tab1, tab2 = st.tabs(["🧠 Live Execution Trace (XAI)", "🕸️ Knowledge Graph"])
    
    # TAB 1: EXPLAINABLE AI TRACE
    with tab1:
        st.subheader("Agent Reasoning & Data Exchange")
        
        if st.session_state.trace_logs:
            for log in st.session_state.trace_logs:
                
                # 1. THOUGHT (LLM Plan)
                if log["type"] == "thought":
                    with st.expander(log["content"], expanded=False):
                        st.caption("Raw LLM Input Prompt:")
                        st.code(log["details"], language="text")

                # 2. ACTION (Tool Call)
                elif log["type"] == "tool_call":
                    st.markdown(log["message"])
                    with st.expander(f"📥 Input to {log['tool']}", expanded=True):
                        st.json(log["input"])

                # 3. OBSERVATION (Tool Result)
                elif log["type"] == "tool_result":
                    with st.expander("📤 Output from Tool", expanded=True):
                        # Try to format JSON output if possible, else text
                        try:
                            st.json(log["output"])
                        except:
                            st.markdown(f"```\n{str(log['output'])}\n```")
                
                # 4. DECISION LOG
                elif log["type"] == "action":
                    st.info(log["content"])

        else:
            st.info("Waiting for agent execution... Ask a question to see the live trace.")

    # TAB 2: GRAPH VIEW
    with tab2:
        st.subheader("Graph View")
        try:
            results = graph.query("""
            MATCH (n)-[r]->(m) 
            RETURN n.id AS source, type(r) AS type, m.id AS target LIMIT 50
            """)
            nodes = []
            edges = []
            node_ids = set()
            for record in results:
                if record['source'] not in node_ids:
                    nodes.append(Node(id=record['source'], label=record['source'], size=15, color="#FF6F61"))
                    node_ids.add(record['source'])
                if record['target'] not in node_ids:
                    nodes.append(Node(id=record['target'], label=record['target'], size=15, color="#6B5B95"))
                    node_ids.add(record['target'])
                edges.append(Edge(source=record['source'], target=record['target'], label=record['type']))
            
            if nodes:
                config = Config(width=600, height=500, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.write("Graph is empty. Upload a PDF to populate.")
        except Exception:
            st.write("No graph data to display.")