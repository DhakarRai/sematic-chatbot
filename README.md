# 🤖 Company Knowledge Chatbot

A semantic search chatbot that answers questions from your company data using AI embeddings and FAISS.

---

## ✨ Features

- **Semantic Search** – Understands meaning, not just keywords
- **Fast Response** – FAISS vector search (~30ms per query)
- **No Hallucination** – Only returns answers from your data
- **Beautiful UI** – Modern chat interface included
- **CPU Friendly** – No GPU required

---

## 🏗️ Architecture

```
Your Data (book.txt)
       ↓
   Chunking (split by lines)
       ↓
   Embedding (all-MiniLM-L6-v2)
       ↓
   FAISS Vector Database
       ↓
   FastAPI Server
       ↓
   Web Chat Interface
```

---

## 📁 Project Structure

```
company chat bot/
├── app/
│   ├── main.py           # FastAPI server + routes
│   ├── embedding.py      # Sentence transformer model
│   ├── faiss_store.py    # FAISS save/load/search
│   ├── utils.py          # Text chunking
│   └── templates/
│       └── index.html    # Chat UI
├── data/
│   └── book.txt          # Your company data
├── faiss_index/
│   ├── index.faiss       # Vector index
│   └── chunks.pkl        # Text chunks
├── build_index.py        # Build FAISS index
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies (First time only)
```bash
cd "company chat bot"
pip install -r requirements.txt
```

### 2. Add Your Data
Edit `data/book.txt` with your company information:
```
Our shop name is ABC Store.
We are located at MG Road, Bangalore.
We sell rice at ₹60 per kg.
We are open from 9 AM to 9 PM.
Contact number is 9876543210.
```

### 3. Build the Index
```bash
python build_index.py
```

### 4. Run the Server

**Development (with auto-reload):**
```bash
uvicorn app.main:app --reload
```

**Production (with 4 workers):**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Open in Browser
Visit: **http://127.0.0.1:8000**

---

## 🧪 How to Test

### API Endpoints

| URL | Method | Purpose |
|-----|--------|---------|
| http://127.0.0.1:8000 | GET | Chat UI |
| http://127.0.0.1:8000/chat | POST | Chat API |
| http://127.0.0.1:8000/health | GET | Health + Cache stats |
| http://127.0.0.1:8000/cache/clear | GET | Clear cache |

### Test Questions

| Question | Expected Answer |
|----------|-----------------|
| Where is your shop? | MG Road, Bangalore |
| Rice price? | ₹60 per kg |
| Store hours? | 9 AM to 9 PM |
| Phone number? | 9876543210 |
| Shop name? | ABC Store |

### Test Caching

1. Ask: **"Where is your shop?"** → Check `response_time_ms` (~30ms)
2. Ask same question again → Check `response_time_ms` (~1-2ms) ⚡

### API Testing (curl)

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "rice price"}'
```

Response:
```json
{
  "question": "rice price",
  "answer": "We sell rice at ₹60 per kg.",
  "cached": false,
  "response_time_ms": 28.5
}
```

### Test Checklist

- ☐ Server starts without error
- ☐ Chat UI loads
- ☐ Questions get answers
- ☐ Different questions = different answers
- ☐ Same question = cached (faster)
- ☐ Health endpoint works

---

## 🔧 How It Works

1. **Chunking** – Each line in `book.txt` becomes a separate chunk
2. **Embedding** – AI model converts chunks to meaning vectors
3. **FAISS Index** – Vectors stored for fast similarity search
4. **Query** – User question → embedding → find closest chunk
5. **Response** – Return the matching text

---

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Embedding | sentence-transformers |
| Vector DB | FAISS |
| Frontend | HTML/CSS/JS |
| Model | all-MiniLM-L6-v2 |

---

## 🧠 AI Model Details

This chatbot uses **`all-MiniLM-L6-v2`** from Sentence Transformers.

### Model Specifications

| Property | Value |
|----------|-------|
| Model Name | all-MiniLM-L6-v2 |
| Provider | Hugging Face / Sentence Transformers |
| Vector Dimensions | 384 |
| Max Tokens | 256 |
| Size | ~80 MB |
| Speed | Very Fast |
| GPU Required | ❌ No (CPU works great) |

### Why This Model?

✅ **Lightweight** – Only 80MB, loads quickly  
✅ **Fast** – Encodes text in milliseconds  
✅ **Accurate** – Great for semantic similarity  
✅ **Free** – Open source, no API costs  
✅ **Offline** – Works without internet after download  

### What It Does

```
Text: "Where is your shop?"
         ↓
   AI Model Processing
         ↓
Vector: [0.23, -0.11, 0.88, ...] (384 numbers)
```

The model converts text into **meaning vectors** (numbers that represent meaning).
Similar meanings = similar vectors = accurate matching!

### Alternative Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| all-MiniLM-L6-v2 ✅ | 80MB | ⚡ Fast | Good |
| all-mpnet-base-v2 | 420MB | Medium | Better |
| paraphrase-multilingual | 1GB | Slow | Multilingual |

We use **MiniLM** for the best balance of speed and accuracy.

---

## ⚡ Performance

- 1 user: ~30ms
- 1000 users: ~120ms
- Runs on CPU only

---

## 📝 Adding More Data

1. Edit `data/book.txt` (one fact per line)
2. Run `python build_index.py`
3. Restart the server

---

## 🎯 Best For

- Company FAQ bots
- Shop information assistants
- Product query systems
- Help desk automation

---

## ❗ Important Note (AI Scope)

This chatbot is **NOT** a generative AI like ChatGPT.

❌ It does **not**:
- Invent answers
- Chat freely
- Give opinions

✅ It **only**:
- Searches company data
- Finds the closest matching information
- Returns exact stored text

This ensures:
- ✅ High accuracy
- ✅ No fake information
- ✅ Full control over answers

---

## 🤔 Why Not Rasa?

Rasa is designed for conversational workflows (intents, dialogs).
This project focuses on **large document understanding** and **semantic search**.

For company knowledge bots, embedding-based search is:
- ⚡ Faster
- 🔧 Easier to maintain
- 📈 More scalable

| Feature | This Bot | Rasa |
|---------|----------|------|
| Large documents | ✅ Easy | ❌ Hard |
| Setup time | ✅ Minutes | ❌ Hours |
| Intent training | ✅ Not needed | ❌ Manual |
| Semantic understanding | ✅ Built-in | ❌ Limited |

---

Made with ❤️ for simple, accurate company chatbots.
