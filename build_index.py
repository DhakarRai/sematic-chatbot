from app.utils import chunk_text
from app.embedding import embed_text
from app.faiss_store import save_faiss

with open("data/book.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text)
vectors = embed_text(chunks)

save_faiss(vectors, chunks)
print("FAISS index built successfully")
