
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# ==========================================
#  híbrido Search: The "Best of Both Worlds"
# ==========================================
# Scenario: A Technical Support Knowledge Base.
#
# THE PROBLEM:
# Users often search for specific error codes (e.g., "Error 998877").
# - Vector Search (Semantic) might fail because "998877" is just a number; it might map it to "random numbers" or generic "errors".
# - Keyword Search (BM25) is great for "998877", but fails if the user asks "My screen is frozen" (semantic concept).
#
# THE SOLUTION:
# Combine them! 
# Score = (Vector_Score * Alpha) + (Keyword_Score * (1 - Alpha))

def main():
    # --- 1. THE DATASET ---
    documents = [
        "Doc 1: How to fix a frozen screen or unresponsive generic computer error.",  # General concept
        "Doc 2: Network connectivity and Wi-Fi troubleshooting guide.",             # Different topic
        "Doc 3: Error 998877: Critical Database Corruption. Immediate patch required.", # The target for ID search
        "Doc 4: Understanding system error logs and random hex codes.",             # Confusing distractor
        "Doc 5: Holiday calendar for 2024."                                         # Irrelevant
    ]

    # --- 2. THE QUERY ---
    # A mix of concept ("How to fix") and precise keyword ("Error 998877")
    query = "How to fix Error 998877"
    
    print(f"🔎 QUERY: '{query}'")
    print("-" * 60)

    # --- 3. DENSE SEARCH (Vector Embeddings) ---
    print("🧠 initializing Vector Model (Semantic Search)...")
    # Load a lightweight, efficient model
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    # Encode docs and query
    doc_embeddings = model.encode(documents)
    query_embedding = model.encode(query)
    
    # Calculate Cosine Similarity (Scores are 0.0 to 1.0)
    vector_scores = util.cos_sim(query_embedding, doc_embeddings)[0].numpy()

    # --- 4. SPARSE SEARCH (BM25 Keywords) ---
    print("📚 Initializing BM25 (Keyword Search)...")
    # Tokenize (simple splitting for demo)
    tokenized_corpus = [doc.lower().split() for doc in documents]
    tokenized_query = query.lower().split()
    
    bm25 = BM25Okapi(tokenized_corpus)
    keyword_scores = np.array(bm25.get_scores(tokenized_query))
    
    # --- 5. NORMALIZATION ---
    # BM25 scores are unbounded (can be 0, 10, 20...). We MUST normalize them to 0-1 range to match Vector scores.
    def normalize(scores):
        if np.max(scores) == np.min(scores):
            return scores # Avoid division by zero
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    norm_keyword_scores = normalize(keyword_scores)

    # --- 6. HYBRID FUSION ---
    # Let's try different Alpha values to see the magic
    
    alphas = [1.0, 0.0, 0.5] # Semantic only, Keyword only, Hybrid
    
    for alpha in alphas:
        if alpha == 1.0:
            mode = "PURE VECTOR (Semantic)"
        elif alpha == 0.0:
            mode = "PURE KEYWORD (BM25)"
        else:
            mode = "HYBRID (Mixed)"
            
        print(f"\n--- MODE: {mode} (Alpha={alpha}) ---")
        
        # The Hybrid Formula
        hybrid_scores = (vector_scores * alpha) + (norm_keyword_scores * (1 - alpha))
        
        # Sort results
        # Get indices of top scores
        top_indices = np.argsort(hybrid_scores)[::-1]
        
        for rank, idx in enumerate(top_indices):
            print(f"Rank {rank+1}: {documents[idx]}")
            print(f"   Score: {hybrid_scores[idx]:.4f} | Vector: {vector_scores[idx]:.4f} | Keyword: {norm_keyword_scores[idx]:.4f}")

if __name__ == "__main__":
    main()
