from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.embedding import model
from app.faiss_store import load_faiss, search_top_k
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
import os
import time
import json
import re

app = FastAPI()

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

# Load FAISS index once at startup
index, chunks = load_faiss()

# Load config once at startup
def load_config():
    config_path = os.path.join(DATA_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# Thread pool for CPU-bound tasks (embedding)
executor = ThreadPoolExecutor(max_workers=4)

# Cache stats
cache_stats = {"hits": 0, "misses": 0}

# ============================================
# CONFIGURATION - THRESHOLDS
# ============================================
SIMILARITY_THRESHOLD = 0.20       # Minimum cosine similarity for confident answer
STRICT_THRESHOLD = 0.22           # Higher threshold for short queries (< 3 words)
TOP_K = 5                         # Number of results to retrieve

# ============================================
# GREETING HANDLING - Bypass FAISS completely
# ============================================
GREETING_PATTERNS = {
    "hi", "hii", "hiii", "hello", "helo", "hellow", "hey", "heya",
    "good morning", "good afternoon", "good evening", "good night",
    "morning", "afternoon", "evening", "night",
    "namaste", "namaskar", "namaskaram",
    "howdy", "greetings", "sup", "wassup", "whatsup", "yo",
    "hai", "haii", "hlo", "hlw"
}

GREETING_RESPONSE = "Hello 😊 Welcome to Nova – My Mentor, your trusted digital learning companion. How can I help you today?"

FALLBACK_RESPONSE = "I don't have specific information about that in my knowledge base. To give you accurate guidance, I can connect you with our support team. Would you like that?"

CLARIFICATION_RESPONSE = "Could you please provide more details about what you'd like to know about Nova – My Mentor?"

# Topics that are definitely NOT covered in Nova My Mentor knowledge base
UNRELATED_TOPICS = {
    # Programming/Tech not covered
    "coding", "programming", "python", "java", "javascript", "html", "css", "c++",
    "software", "website", "app development", "machine learning", "ai course",
    # General knowledge not covered
    "weather", "news", "cricket", "football", "movie", "song", "music",
    "recipe", "cooking", "food", "restaurant",
    # Other education not covered
    "college", "university", "degree", "mbbs", "engineering entrance", "neet",
    "upsc", "ssc", "bank exam", "government job",
    # Random
    "joke", "story", "game", "play"
}

def is_unrelated_topic(text: str) -> bool:
    """Check if query is about a topic definitely not in our knowledge base."""
    text_lower = text.lower()
    for topic in UNRELATED_TOPICS:
        if topic in text_lower:
            return True
    return False

def is_greeting(text: str) -> bool:
    """
    Check if input is a greeting - must bypass FAISS completely.
    """
    text_lower = text.lower().strip()
    
    # Remove all punctuation and extra spaces
    text_clean = re.sub(r'[^\w\s]', '', text_lower).strip()
    text_clean = re.sub(r'\s+', ' ', text_clean)
    
    # Exact match check
    if text_clean in GREETING_PATTERNS:
        return True
    
    # Check for greeting at start with short total length
    words = text_clean.split()
    if len(words) <= 3:
        for greeting in GREETING_PATTERNS:
            if text_clean.startswith(greeting):
                return True
    
    return False

# ============================================
# SMART ANSWER SELECTION
# ============================================
def select_best_answer(results: list, query: str) -> tuple:
    """
    Select the best answer from top-k results with strict confidence check.
    
    Args:
        results: List of (chunk, similarity_score) tuples, sorted by score desc
        query: Original user query
        
    Returns:
        tuple: (answer, confidence_score, is_confident)
    """
    if not results:
        return FALLBACK_RESPONSE, 0.0, False
    
    # Determine threshold based on query length
    query_words = len(query.split())
    threshold = STRICT_THRESHOLD if query_words < 3 else SIMILARITY_THRESHOLD
    
    # Get best result
    best_chunk, best_score = results[0]
    
    # STRICT CHECK: If below threshold, return fallback - NO EXCEPTIONS
    if best_score < threshold:
        return FALLBACK_RESPONSE, best_score, False
    
    # From confident results, select the most informative answer
    confident_results = [(chunk, score) for chunk, score in results if score >= threshold * 0.95]
    
    if not confident_results:
        return FALLBACK_RESPONSE, best_score, False
    
    # Score each candidate for quality
    best_answer = best_chunk
    best_quality_score = 0
    
    for chunk, score in confident_results:
        quality_score = score
        
        # Bonus for longer, more complete answers (up to 0.05)
        if len(chunk) > 80:
            quality_score += 0.03
        if len(chunk) > 150:
            quality_score += 0.02
        
        # Bonus for explanatory content
        explanatory_words = ['is', 'are', 'helps', 'provides', 'supports', 'includes', 'offers', 'designed', 'can', 'allows']
        if any(word in chunk.lower() for word in explanatory_words):
            quality_score += 0.02
        
        # Penalty for header-like or incomplete content
        if chunk.strip().endswith(':'):
            quality_score -= 0.1
        if len(chunk.split()) < 6:
            quality_score -= 0.05
        
        if quality_score > best_quality_score:
            best_quality_score = quality_score
            best_answer = chunk
    
    return best_answer, best_score, True

def smart_search(question: str) -> tuple:
    """
    Production-safe semantic search with:
    1. Greeting bypass (no FAISS)
    2. Unrelated topic detection
    3. Top-K retrieval
    4. Strict similarity threshold
    5. Mandatory fallback for low confidence
    
    Returns:
        tuple: (answer, was_greeting, confidence, is_confident)
    """
    # 1. GREETING CHECK - Bypass FAISS completely
    if is_greeting(question):
        return GREETING_RESPONSE, True, 1.0, True
    
    # 2. UNRELATED TOPIC CHECK - Return fallback for topics not in knowledge base
    if is_unrelated_topic(question):
        print(f"⛔ Unrelated topic detected: '{question}'")
        return FALLBACK_RESPONSE, False, 0.0, False
    
    # 3. Very short query check
    if len(question.strip()) < 2:
        return CLARIFICATION_RESPONSE, False, 0.0, False
    
    # 4. Encode query and search top-k
    query_vector = model.encode(question)
    results = search_top_k(query_vector, index, chunks, k=TOP_K)
    
    # DEBUG: Print top results with scores
    print(f"\n🔍 DEBUG - Query: '{question}'")
    print(f"📊 Top {len(results)} results:")
    for i, (chunk, score) in enumerate(results[:3]):
        preview = chunk[:80].replace('\n', ' ')
        print(f"   {i+1}. Score: {score:.4f} | {preview}...")
    
    # 4. Select best answer with strict confidence check
    answer, confidence, is_confident = select_best_answer(results, question)
    
    print(f"✅ Selected: confidence={confidence:.4f}, is_confident={is_confident}")
    
    return answer, False, confidence, is_confident

class Question(BaseModel):
    question: str

@lru_cache(maxsize=1000)
def cached_smart_search(question: str) -> tuple:
    """Cache smart search results."""
    return smart_search(question)

def get_answer(question: str) -> tuple:
    """Get answer with cache tracking."""
    cache_info = cached_smart_search.cache_info()
    
    answer, was_greeting, confidence, is_confident = cached_smart_search(question.lower().strip())
    
    new_cache_info = cached_smart_search.cache_info()
    was_cached = new_cache_info.hits > cache_info.hits
    
    if was_cached:
        cache_stats["hits"] += 1
    else:
        cache_stats["misses"] += 1
    
    return answer, was_cached, confidence, is_confident

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the chatbot HTML page"""
    with open(os.path.join(TEMPLATES_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/config")
async def get_config():
    """Get UI configuration from config.json"""
    return config

@app.post("/chat")
async def chat(q: Question):
    """
    Production-safe chat endpoint with:
    - Greeting bypass (no FAISS for greetings)
    - Top-5 retrieval with cosine similarity
    - Strict similarity threshold (0.65-0.70)
    - Mandatory fallback for low confidence
    """
    start_time = time.time()
    
    loop = asyncio.get_event_loop()
    answer, was_cached, confidence, is_confident = await loop.run_in_executor(
        executor, 
        get_answer, 
        q.question
    )
    
    response_time = round((time.time() - start_time) * 1000, 2)
    
    # Print to terminal for debugging
    print(f"\n{'='*50}")
    print(f"👤 USER: {q.question}")
    print(f"🤖 BOT:  {answer[:100]}..." if len(answer) > 100 else f"🤖 BOT:  {answer}")
    print(f"📊 Confidence: {confidence:.3f} | Confident: {is_confident} | Cached: {was_cached} | Time: {response_time}ms")
    print(f"{'='*50}\n")

    return {
        "question": q.question,
        "answer": answer,
        "confidence": round(confidence, 3),
        "is_confident": is_confident,
        "cached": was_cached,
        "response_time_ms": response_time
    }

@app.get("/health")
async def health():
    """Health check endpoint with feature status"""
    cache_info = cached_smart_search.cache_info()
    return {
        "status": "ok",
        "features": {
            "greeting_bypass": True,
            "top_k_retrieval": TOP_K,
            "similarity_metric": "cosine",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "strict_threshold": STRICT_THRESHOLD,
            "fallback_enabled": True
        },
        "cache": {
            "enabled": True,
            "max_size": 1000,
            "current_size": cache_info.currsize,
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "hit_rate": f"{(cache_stats['hits'] / max(1, cache_stats['hits'] + cache_stats['misses']) * 100):.1f}%"
        }
    }

@app.get("/cache/clear")
async def clear_cache():
    """Clear the cache"""
    cached_smart_search.cache_clear()
    cache_stats["hits"] = 0
    cache_stats["misses"] = 0
    return {"status": "cache cleared"}

@app.get("/config/reload")
async def reload_config():
    """Reload config from file"""
    global config
    config = load_config()
    return {"status": "config reloaded", "config": config}
