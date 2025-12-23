from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.faiss_store import load_faiss
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import time
import json
import re

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# Load FAISS index + chunks (already built locally)
index, chunks = load_faiss()

def load_config():
    with open(os.path.join(DATA_DIR, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

executor = ThreadPoolExecutor(max_workers=2)
cache_stats = {"hits": 0, "misses": 0}

TOP_K = 3

GREETING_PATTERNS = {
    "hi", "hello", "hey", "namaste", "hii", "hai", "hlw"
}

GREETING_RESPONSE = (
    "Hello 😊 Welcome to Nova – My Mentor, your trusted digital learning companion. "
    "How can I help you today?"
)

FALLBACK_RESPONSE = (
    "I don’t have exact information on that. "
    "Would you like me to connect you with our support team?"
)

def is_greeting(text: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return clean in GREETING_PATTERNS

def keyword_search(question: str, chunks: list, limit: int = 3):
    q = question.lower()
    results = []

    for chunk in chunks:
        if q in chunk.lower():
            results.append((chunk, 1.0))

    return results[:limit]

def smart_search(question: str):
    if is_greeting(question):
        return GREETING_RESPONSE, True, 1.0, True

    if len(question.strip()) < 2:
        return "Please ask a clear question 😊", False, 0.0, False

    results = keyword_search(question, chunks, TOP_K)

    if not results:
        return FALLBACK_RESPONSE, False, 0.0, False

    return results[0][0], False, 1.0, True

class Question(BaseModel):
    question: str

@lru_cache(maxsize=500)
def cached_search(q: str):
    return smart_search(q)

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(TEMPLATES_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(q: Question):
    start = time.time()

    loop = asyncio.get_event_loop()
    answer, greeting, confidence, ok = await loop.run_in_executor(
        executor, cached_search, q.question.lower()
    )

    return {
        "question": q.question,
        "answer": answer,
        "confidence": confidence,
        "ok": ok,
        "response_time_ms": round((time.time() - start) * 1000, 2)
    }

@app.get("/health")
async def health():
    return {"status": "ok", "render_safe": True}
