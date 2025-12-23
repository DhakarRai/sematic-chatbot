import faiss
import pickle
import numpy as np

def save_faiss(vectors, chunks):
    """
    Save FAISS index using Inner Product (for cosine similarity).
    Vectors must be normalized before adding.
    """
    dim = vectors.shape[1]
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    
    # Use IndexFlatIP for cosine similarity (Inner Product of normalized vectors = cosine)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, "faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Saved {len(chunks)} chunks with cosine similarity index")

def load_faiss():
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search_top_k(query_vector, index, chunks, k=5):
    """
    Search for top-k similar chunks using cosine similarity.
    
    Returns:
        List of tuples: [(chunk, cosine_similarity_score), ...]
        Score range: -1 to 1 (higher = more similar)
        
    Note: FAISS IndexFlatIP returns inner product scores.
    For normalized vectors, inner product = cosine similarity.
    """
    # Normalize query vector
    query_norm = query_vector / np.linalg.norm(query_vector)
    query_norm = np.array([query_norm]).astype('float32')
    
    # Search returns (scores, indices)
    scores, indices = index.search(query_norm, k)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = scores[0][i]  # This is cosine similarity (0 to 1 for normalized vectors)
        
        if idx >= 0 and idx < len(chunks):
            results.append((chunks[idx], float(score)))
    
    return results

# Keep old search function for backward compatibility
def search(query_vector, index, chunks, k=1):
    results = search_top_k(query_vector, index, chunks, k=k)
    if results:
        return results[0][0]
    return ""
