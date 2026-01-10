"""
Chunking Engine for Agentic RAG Demo
Implements Contextual and Propositional chunking strategies
"""

import os
from typing import List, Dict, Generator
from pypdf import PdfReader
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import config


class ChunkingEngine:
    """Handles dual chunking strategy: Contextual + Propositional"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.chroma_client = None
        self.collection = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def contextual_chunking(self, text: str, document_name: str) -> Generator[Dict, None, None]:
        """
        Contextual Chunking: Adds document context to each chunk
        Yields chunks with metadata for streaming display
        """
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CONTEXTUAL_CHUNK_SIZE,
            chunk_overlap=config.CONTEXTUAL_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Add contextual information to each chunk
        document_context = f"Document: {document_name}\n"
        
        for i, chunk in enumerate(chunks):
            # Add position context
            position_context = f"Section {i+1} of {len(chunks)}\n"
            
            # Combine context with chunk
            contextual_chunk = document_context + position_context + chunk
            
            yield {
                "type": "contextual",
                "chunk_id": f"{document_name}_ctx_{i}",
                "content": contextual_chunk,
                "original": chunk,
                "metadata": {
                    "document": document_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_type": "contextual"
                }
            }
    
    def propositional_chunking(self, text: str, document_name: str) -> Generator[Dict, None, None]:
        """
        Propositional Chunking: Uses LLM to break complex sentences into simple facts
        Yields propositions with metadata for streaming display
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') 
                     if len(p.strip()) > config.PROPOSITION_MIN_PARAGRAPH_LENGTH]
        
        # Prompt template for proposition extraction
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a text analyzer that extracts simple, standalone facts from complex text."),
            ("user", """Read the following text and break it down into simple, standalone propositions (facts).

Rules:
1. Each proposition should be a complete sentence that stands alone
2. Remove complex grammar and nested clauses
3. Each fact should be independently understandable
4. Return as a numbered list

Text:
{paragraph}

Extract the simple propositions:""")
        ])
        
        chain = prompt | self.llm
        
        proposition_id = 0
        for para_idx, paragraph in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs for demo
            try:
                response = chain.invoke({"paragraph": paragraph})
                
                # Parse propositions from response
                propositions = []
                for line in response.content.strip().split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                        # Remove numbering/bullets
                        prop = line.lstrip('0123456789.-•) ').strip()
                        if prop:
                            propositions.append(prop)
                
                # Yield each proposition
                for prop in propositions:
                    yield {
                        "type": "propositional",
                        "chunk_id": f"{document_name}_prop_{proposition_id}",
                        "content": prop,
                        "metadata": {
                            "document": document_name,
                            "paragraph_index": para_idx,
                            "chunk_type": "propositional",
                            "source_paragraph_length": len(paragraph)
                        }
                    }
                    proposition_id += 1
                    
            except Exception as e:
                print(f"Error processing paragraph {para_idx}: {e}")
                continue
    
    def initialize_vector_db(self) -> chromadb.Collection:
        """Initialize ChromaDB collection"""
        # Create persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=config.COLLECTION_NAME)
            print(f"✅ Loaded existing collection: {config.COLLECTION_NAME}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"description": "Agentic RAG Demo - Contextual + Propositional Chunks"}
            )
            print(f"✅ Created new collection: {config.COLLECTION_NAME}")
        
        return self.collection
    
    def add_chunks_to_vectordb(self, chunks: List[Dict]):
        """Add chunks to ChromaDB with embeddings"""
        if not self.collection:
            self.initialize_vector_db()
        
        # Extract content for embedding
        texts = [chunk["content"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
    
    def check_vector_db_exists(self) -> bool:
        """Check if vector database already exists"""
        if not config.VECTOR_DB_PATH.exists():
            return False
        
        try:
            client = chromadb.PersistentClient(
                path=str(config.VECTOR_DB_PATH),
                settings=Settings(anonymized_telemetry=False)
            )
            collection = client.get_collection(name=config.COLLECTION_NAME)
            count = collection.count()
            return count > 0
        except:
            return False
    
    def process_pdfs(self, pdf_paths: List[str]) -> Generator[Dict, None, None]:
        """
        Process PDFs with both chunking strategies
        Yields chunks for streaming display and adds to vector DB
        """
        all_chunks = []
        
        for pdf_path in pdf_paths:
            document_name = Path(pdf_path).stem
            
            yield {
                "status": "processing_document",
                "document": document_name,
                "message": f"📄 Processing {document_name}..."
            }
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            # Contextual chunking
            yield {
                "status": "chunking",
                "strategy": "contextual",
                "message": "🔍 Creating contextual chunks..."
            }
            
            contextual_chunks = list(self.contextual_chunking(text, document_name))
            for chunk in contextual_chunks[:config.CHUNKING_DISPLAY_LIMIT]:
                yield chunk
                all_chunks.append(chunk)
            
            # Add remaining chunks without display
            all_chunks.extend(contextual_chunks[config.CHUNKING_DISPLAY_LIMIT:])
            
            # Propositional chunking
            yield {
                "status": "chunking",
                "strategy": "propositional",
                "message": "🎯 Extracting propositions with LLM..."
            }
            
            propositional_chunks = list(self.propositional_chunking(text, document_name))
            for chunk in propositional_chunks[:config.CHUNKING_DISPLAY_LIMIT]:
                yield chunk
                all_chunks.append(chunk)
            
            # Add remaining chunks without display
            all_chunks.extend(propositional_chunks[config.CHUNKING_DISPLAY_LIMIT:])
        
        # Initialize vector DB and add all chunks
        yield {
            "status": "vectorizing",
            "message": f"💾 Creating vector database with {len(all_chunks)} chunks..."
        }
        
        self.initialize_vector_db()
        
        # Add chunks in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            self.add_chunks_to_vectordb(batch)
            
        yield {
            "status": "complete",
            "message": f"✅ Vector database created with {len(all_chunks)} chunks!",
            "total_chunks": len(all_chunks)
        }
    
    def retrieve_documents(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant documents from vector DB"""
        if not self.collection:
            self.initialize_vector_db()
        
        if top_k is None:
            top_k = config.TOP_K_RETRIEVE
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return documents
