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

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# Load FAISS index once at startup (already built locally)
index, chunks = load_faiss()

# Load config once at startup
def load_config():
    config_path = os.path.join(DATA_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# Thread pool (safe size for Render Free)
executor = ThreadPoolExecutor(max_workers=2)

# Cache stats
cache_stats = {"hits": 0, "misses": 0}

# ============================================
# CONFIGURATION - THRESHOLDS (KEPT AS IS)
# ============================================
SIMILARITY_THRESHOLD = 0.20
STRICT_THRESHOLD = 0.22
TOP_K = 5

# ============================================
# GREETING HANDLING
# ============================================
GREETING_PATTERNS = {
    "hi", "hii", "hiii", "hello", "helo", "hellow", "hey", "heya",
    "good morning", "good afternoon", "good evening", "good night",
    "morning", "afternoon", "evening", "night",
    "namaste", "namaskar", "namaskaram",
    "howdy", "greetings", "sup", "wassup", "whatsup", "yo",
    "hai", "haii", "hlo", "hlw"
}

GREETING_RESPONSE = (
    "Hello 😊 Welcome to Nova – My Mentor, your trusted digital learning companion. "
    "How can I help you today?"
)

FALLBACK_RESPONSE = (
    "I don't have specific information about that in my knowledge base. "
    "To give you accurate guidance, I can connect you with our support team. "
    "Would you like that?"
)

CLARIFICATION_RESPONSE = (
    "Could you please provide more details about what you'd like to know about Nova – My Mentor?"
)

# Topics not covered
UNRELATED_TOPICS = {
    "coding", "programming", "python", "java", "javascript", "html", "css", "c++",
    "software", "website", "app development", "machine learning", "ai course",
    "weather", "news", "cricket", "football", "movie", "song", "music",
    "recipe", "cooking", "food", "restaurant",
    "college", "university", "degree", "mbbs", "engineering entrance", "neet",
    "upsc", "ssc", "bank exam", "government job",
    "joke", "story", "game", "play"
}

def is_unrelated_topic(text: str) -> bool:
    text_lower = text.lower()
    return any(topic in text_lower for topic in UNRELATED_TOPICS)

def is_greeting(text: str) -> bool:
    text_clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    text_clean = re.sub(r"\s+", " ", text_clean)
    if text_clean in GREETING_PATTERNS:
        return True
    words = text_clean.split()
    if len(words) <= 3:
        return any(text_clean.startswith(g) for g in GREETING_PATTERNS)
    return False

# ============================================
# SAFE SEARCH (NO EMBEDDINGS – RENDER FREE)
# ============================================
def keyword_search(question: str, limit: int = 3):
    q = question.lower()
    matches = []
    for chunk in chunks:
        if q in chunk.lower():
            matches.append((chunk, 1.0))
    return matches[:limit]

def smart_search(question: str) -> tuple:
    if is_greeting(question):
        return GREETING_RESPONSE, True, 1.0, True

    if is_unrelated_topic(question):
        return FALLBACK_RESPONSE, False, 0.0, False

    if len(question.strip()) < 2:
        return CLARIFICATION_RESPONSE, False, 0.0, False

    results = keyword_search(question, TOP_K)

    if not results:
        return FALLBACK_RESPONSE, False, 0.0, False

    best_chunk, score = results[0]
    return best_chunk, False, score, True

class Question(BaseModel):
    question: str

@lru_cache(maxsize=1000)
def cached_smart_search(question: str):
    return smart_search(question)

def get_answer(question: str):
    before = cached_smart_search.cache_info().hits
    answer, was_greeting, confidence, is_confident = cached_smart_search(question.lower().strip())
    after = cached_smart_search.cache_info().hits

    if after > before:
        cache_stats["hits"] += 1
    else:
        cache_stats["misses"] += 1

    return answer, after > before, confidence, is_confident

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(TEMPLATES_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/config")
async def get_config():
    return config

@app.post("/chat")
async def chat(q: Question):
    start = time.time()
    loop = asyncio.get_event_loop()
    answer, cached, confidence, confident = await loop.run_in_executor(
        executor, get_answer, q.question
    )

    return {
        "question": q.question,
        "answer": answer,
        "confidence": round(confidence, 3),
        "is_confident": confident,
        "cached": cached,
        "response_time_ms": round((time.time() - start) * 1000, 2)
    }

@app.get("/health")
async def health():
    cache_info = cached_smart_search.cache_info()
    return {
        "status": "ok",
        "render_safe": True,
        "cache": {
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "size": cache_info.currsize
        }
    }

@app.get("/cache/clear")
async def clear_cache():
    cached_smart_search.cache_clear()
    cache_stats["hits"] = 0
    cache_stats["misses"] = 0
    return {"status": "cache cleared"}
