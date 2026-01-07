from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- STEP 1: OUR DATA (The Knowledge Base) ---
documents = [
    "General Guide: How to fix slow Wi-Fi and internet connection.",
    "Log file entry: System check passed. Error 5053 not found.",
    "Troubleshooting Guide: Fixing Internet connection failure (Error 5053)."
]

# The user's search query
query = "internet down error 5053"

# --- STEP 2: REAL KEYWORD SEARCH (BM25) ---
# We break the documents into individual words ("tokens")
tokenized_corpus = [doc.lower().split(" ") for doc in documents]
tokenized_query = query.lower().split(" ")

# Create the BM25 object (The "Keyword Judge")
bm25 = BM25Okapi(tokenized_corpus)

# Get raw keyword scores
keyword_scores = bm25.get_scores(tokenized_query)

# --- STEP 3: REAL VECTOR SEARCH (Embeddings) ---
# Load a small, free AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Turn documents and query into numbers (Vectors)
doc_embeddings = model.encode(documents)
query_embedding = model.encode(query)

# Calculate similarity (The "Vector Judge")
# This returns a score between 0 and 1
vector_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# --- STEP 4: NORMALIZATION (Crucial Step!) ---
# BM25 scores can be anything (e.g., 0 to 15). Vector scores are 0 to 1.
# We must squash BM25 scores to be between 0 and 1 so we can mix them fairly.
def normalize(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

norm_keyword_scores = normalize(keyword_scores)
# Vector scores are usually already normalized enough for this simple example

# --- STEP 5: HYBRID BLEND ---
alpha = 0.5  # Equal weight to both

# The Formula: (Vector * Alpha) + (Keyword * (1 - Alpha))
hybrid_scores = (vector_scores * alpha) + (norm_keyword_scores * (1 - alpha))

# --- PRINT RESULTS ---
print(f"Query: '{query}'\n")
for i, doc in enumerate(documents):
    print(f"Doc {i+1}: {doc}")
    print(f"   - Vector Score: {vector_scores[i]:.4f}")
    print(f"   - Keyword Score (Norm): {norm_keyword_scores[i]:.4f}")
    print(f"   - HYBRID SCORE: {hybrid_scores[i]:.4f}\n")