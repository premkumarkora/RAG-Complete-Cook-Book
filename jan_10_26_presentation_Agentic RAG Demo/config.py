"""
Configuration file for Agentic RAG Demo
Centralizes all model settings, paths, and parameters
"""

import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent
VECTOR_DB_PATH = BASE_DIR / "vector_db"
PDF_DIR = BASE_DIR

# Model Configuration
MODEL_NAME = "gpt-4o-mini"  # Using gpt-4o-mini as gpt-5-nano may not be available
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0  # For deterministic outputs

# Chunking Parameters
CONTEXTUAL_CHUNK_SIZE = 1000
CONTEXTUAL_CHUNK_OVERLAP = 200
PROPOSITION_MIN_PARAGRAPH_LENGTH = 100  # Minimum chars for propositional chunking

# Retrieval Parameters
TOP_K_RETRIEVE = 5  # Number of documents to retrieve
RELEVANCE_THRESHOLD = 0.6  # Minimum relevance score (0-1)

# RAGAS Configuration
RAGAS_METRICS = ["faithfulness", "answer_relevance", "context_precision"]

# UI Configuration
CHUNKING_DISPLAY_LIMIT = 10  # Max chunks to show in real-time
WORKFLOW_ANIMATION_DELAY = 0.5  # Seconds between workflow steps

# Collection name for ChromaDB
COLLECTION_NAME = "agentic_rag_demo"
