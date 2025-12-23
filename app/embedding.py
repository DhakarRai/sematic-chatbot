"""
embedding.py

NOTE:
This file is intentionally DISABLED on Render Free plan.

Reason:
- sentence-transformers + torch exceed 512MB RAM
- Causes Out Of Memory (OOM) on Render Free

FAISS index is already built locally.
This server only performs lightweight lookup.

If you want semantic embeddings:
- Run locally
- Or use a paid server (>= 2GB RAM)
"""

def embed_text(texts):
    """
    Dummy embedding function (Render-safe).

    This exists only to avoid import errors.
    It MUST NOT be used on Render Free.
    """
    raise RuntimeError(
        "Embedding is disabled on Render Free plan. "
        "FAISS index is prebuilt locally."
    )
