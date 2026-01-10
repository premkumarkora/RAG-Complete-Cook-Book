"""
Agentic Workflow using LangGraph
Implements StateGraph with Retrieve → Grade → Generate nodes
"""

from typing import TypedDict, List, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import config


class AgentState(TypedDict):
    """State schema for LangGraph workflow"""
    question: str
    retrieved_docs: List[dict]
    graded_docs: List[dict]
    relevance_scores: List[float]
    query_rewrite: Optional[str]
    final_answer: str
    workflow_history: List[str]
    current_node: str


class AgenticWorkflow:
    """LangGraph-based agentic RAG workflow"""
    
    def __init__(self, chunking_engine):
        self.chunking_engine = chunking_engine
        self.llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
        self.graph = self._build_graph()
    
    def retrieve_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Retrieve relevant documents from vector database
        """
        question = state["question"]
        
        # Add to workflow history
        workflow_history = state.get("workflow_history", [])
        workflow_history.append("🔍 RETRIEVE: Fetching relevant documents from vector database...")
        
        # Retrieve documents
        docs = self.chunking_engine.retrieve_documents(question, top_k=config.TOP_K_RETRIEVE)
        
        workflow_history.append(f"   ✅ Retrieved {len(docs)} documents")
        
        return {
            **state,
            "retrieved_docs": docs,
            "workflow_history": workflow_history,
            "current_node": "retrieve"
        }
    
    def grade_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Grade document relevance using LLM
        This is the critical step that prevents hallucinations
        """
        question = state["question"]
        retrieved_docs = state["retrieved_docs"]
        workflow_history = state.get("workflow_history", [])
        
        workflow_history.append("⚖️ GRADE: Evaluating document relevance with LLM...")
        
        # Grading prompt
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance of retrieved documents to a user question."),
            ("user", """Here is the retrieved document:
{document}

Here is the user question:
{question}

Determine if the document contains information relevant to answering the question.
Answer only 'yes' or 'no'. If the document contains ANY relevant information, answer 'yes'.

Relevance:""")
        ])
        
        grade_chain = grade_prompt | self.llm
        
        graded_docs = []
        relevance_scores = []
        
        for i, doc in enumerate(retrieved_docs):
            try:
                # Grade document
                response = grade_chain.invoke({
                    "document": doc["content"],
                    "question": question
                })
                
                # Parse response
                grade = response.content.strip().lower()
                is_relevant = "yes" in grade
                
                # Get document preview (first 200 characters)
                doc_preview = doc["content"][:200].replace('\n', ' ') + "..."
                
                if is_relevant:
                    graded_docs.append(doc)
                    relevance_scores.append(1.0)
                    workflow_history.append(f"   ✅ Doc {i+1}: RELEVANT")
                    workflow_history.append(f"      📄 Content: {doc_preview}")
                else:
                    relevance_scores.append(0.0)
                    workflow_history.append(f"   ❌ Doc {i+1}: NOT RELEVANT")
                    workflow_history.append(f"      📄 Content: {doc_preview}")
                    
            except Exception as e:
                workflow_history.append(f"   ⚠️ Doc {i+1}: Error grading - {str(e)}")
                relevance_scores.append(0.0)
        
        # Summary
        relevant_count = len(graded_docs)
        workflow_history.append(f"   📊 Result: {relevant_count}/{len(retrieved_docs)} documents are relevant")
        
        return {
            **state,
            "graded_docs": graded_docs,
            "relevance_scores": relevance_scores,
            "workflow_history": workflow_history,
            "current_node": "grade"
        }
    
    def generate_node(self, state: AgentState) -> AgentState:
        """
        Node 3: Generate answer using only graded (relevant) documents
        """
        question = state["question"]
        graded_docs = state["graded_docs"]
        workflow_history = state.get("workflow_history", [])
        
        workflow_history.append("✍️ GENERATE: Creating answer from relevant documents...")
        
        if not graded_docs:
            # No relevant documents found
            final_answer = "❌ NO DATA: I couldn't find any relevant information in the knowledge base to answer your question. The retrieved documents were not relevant to your query."
            workflow_history.append("   ⚠️ No relevant documents - returning NO DATA response")
        else:
            # Combine graded documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc['content']}" 
                for i, doc in enumerate(graded_docs)
            ])
            
            # Generation prompt
            generate_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant for question-answering tasks. 
Use ONLY the following retrieved context to answer the question. 
If you cannot answer the question based on the context, say so.
Do NOT use any external knowledge or make assumptions.
Be concise and clear in your response."""),
                ("user", """Context:
{context}

Question: {question}

Answer:""")
            ])
            
            generate_chain = generate_prompt | self.llm
            
            try:
                response = generate_chain.invoke({
                    "context": context,
                    "question": question
                })
                final_answer = response.content.strip()
                workflow_history.append(f"   ✅ Generated answer using {len(graded_docs)} relevant documents")
            except Exception as e:
                final_answer = f"❌ Error generating answer: {str(e)}"
                workflow_history.append(f"   ⚠️ Error: {str(e)}")
        
        return {
            **state,
            "final_answer": final_answer,
            "workflow_history": workflow_history,
            "current_node": "generate"
        }
    
    def decide_to_generate(self, state: AgentState) -> Literal["generate", "no_data"]:
        """
        Conditional edge: Decide whether to generate answer or handle no data
        """
        graded_docs = state.get("graded_docs", [])
        
        if len(graded_docs) > 0:
            return "generate"
        else:
            # Could implement query rewrite here
            return "no_data"
    
    def handle_no_data(self, state: AgentState) -> AgentState:
        """
        Handle case when no relevant documents are found
        Could implement query rewriting here
        """
        workflow_history = state.get("workflow_history", [])
        workflow_history.append("🔄 NO DATA PATH: No relevant documents found")
        
        final_answer = """❌ **NO RELEVANT DATA FOUND**

I searched the knowledge base but couldn't find information relevant to your question. 

**Suggestions:**
- Try rephrasing your question
- Ask about topics covered in the ITC financial reports
- Check if your question relates to the available documents"""
        
        return {
            **state,
            "final_answer": final_answer,
            "workflow_history": workflow_history,
            "current_node": "no_data"
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph"""
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade", self.grade_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("no_data", self.handle_no_data)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        
        # Conditional edge from grade
        workflow.add_conditional_edges(
            "grade",
            self.decide_to_generate,
            {
                "generate": "generate",
                "no_data": "no_data"
            }
        )
        
        # End edges
        workflow.add_edge("generate", END)
        workflow.add_edge("no_data", END)
        
        return workflow.compile()
    
    def run(self, question: str) -> AgentState:
        """
        Run the agentic workflow for a given question
        Returns final state with answer and workflow history
        """
        # Initialize state
        initial_state = {
            "question": question,
            "retrieved_docs": [],
            "graded_docs": [],
            "relevance_scores": [],
            "query_rewrite": None,
            "final_answer": "",
            "workflow_history": [f"❓ Question: {question}"],
            "current_node": "start"
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
