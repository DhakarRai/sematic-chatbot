"""
faiss_store.py

Render Free SAFE version.

- FAISS index is BUILT LOCALLY (VS Code)
- NO embeddings computed on server
- NO index training on server
- ONLY loads prebuilt FAISS index and chunks

Cosine similarity logic is preserved for future paid upgrade.
"""

import faiss
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_DIR = os.path.join(os.path.dirname(BASE_DIR), "faiss_index")

INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(FAISS_DIR, "chunks.pkl")

# ======================================================
# ❌ DISABLED ON RENDER FREE (TRAINING)
# ======================================================
def save_faiss(vectors, chunks):
    """
    Disabled on Render Free.

    FAISS index MUST be built locally.
    """
    raise RuntimeError(
        "save_faiss() is disabled on Render Free. "
        "Build FAISS index locally using VS Code."
    )

# ======================================================
# ✅ SAFE: LOAD PREBUILT FAISS
# ======================================================
def load_faiss():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks

# ======================================================
# ❌ DISABLED ON RENDER FREE (EMBEDDING SEARCH)
# ======================================================
def search_top_k(query_vector, index, chunks, k=5):
    """
    Disabled on Render Free.

    Requires embeddings → not allowed on 512MB RAM.
    """
    raise RuntimeError(
        "search_top_k() requires embeddings and is disabled on Render Free."
    )

# ======================================================
# ❌ DISABLED (BACKWARD COMPATIBILITY)
# ======================================================
def search(query_vector, index, chunks, k=1):
    """
    Disabled wrapper.
    """
    raise RuntimeError(
        "search() requires embeddings and is disabled on Render Free."
    )
