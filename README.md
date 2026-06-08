# 🔍 RAG Q&A Bot — ML Knowledge Base

A production-ready **Retrieval-Augmented Generation (RAG)** application that answers questions about ML articles using FAISS semantic search and the Llama-3.2-3B model via HuggingFace Inference API.

---

## 🏗️ Architecture

```
app.py              ← Streamlit UI layer
├── retriever.py    ← FAISS semantic search (cosine similarity)
├── embedder.py     ← Sentence-Transformer query encoding
└── llm_client.py   ← HuggingFace LLM client + prompt builder
```

**RAG Pipeline:**
1. User submits a question via Streamlit
2. Query is embedded using `all-MiniLM-L6-v2`
3. Top-K most similar chunks are retrieved via FAISS
4. A structured prompt is built with context + question
5. `Llama-3.2-3B-Instruct` generates a cited answer

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- A [HuggingFace account](https://huggingface.co/) with API token access

### 1. Clone the repository
```bash
git clone https://github.com/shrey3108/ML_ChatModel.git
cd ML_ChatModel
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your HuggingFace token
```

`.env` file:
```
HUGGINGFACE_TOKEN=hf_your_token_here
```

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t rag-qa-bot .

# Run
docker run -p 8501:8501 --env-file .env rag-qa-bot
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## ✨ Features

| Feature | Details |
|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim, L2-normalised) |
| **Vector Index** | FAISS `IndexFlatIP` (cosine similarity) |
| **LLM** | `meta-llama/Llama-3.2-3B-Instruct` via HuggingFace API |
| **UI** | Streamlit with configurable Top-K and max tokens |
| **Caching** | `@st.cache_resource` — models loaded once per session |
| **Error Handling** | Input validation, API error handling, graceful fallbacks |
| **Security** | Non-root Docker user, `.env` for secrets, input sanitisation |
| **Tests** | pytest unit tests for retriever and embedder modules |
| **Deployment** | Multi-stage Dockerfile + devcontainer support |

---

## 📁 Project Structure

```
ML_ChatModel/
├── app.py              # Streamlit UI
├── retriever.py        # FAISS retrieval module
├── embedder.py         # Embedding module
├── llm_client.py       # LLM client + prompt builder
├── chunks.json         # Precomputed text chunks
├── embeddings.npy      # Precomputed embedding vectors
├── article1.txt.txt    # Source article 1
├── article2.txt.txt    # Source article 2
├── Dockerfile          # Multi-stage production Dockerfile
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
├── tests/
│   ├── __init__.py
│   └── test_retriever.py  # Unit tests
└── .devcontainer/      # VS Code devcontainer config
```

---

## 🔒 Security Notes

- API tokens are loaded from `.env` — **never hardcode tokens in source code**
- The Docker image runs as a non-root user (`appuser`)
- User queries are sanitised and length-limited before processing
- `.env` is excluded from version control via `.gitignore`

---

## 📊 Performance

- **Time complexity (search):** O(N·D) for `IndexFlatIP` where N = number of chunks, D = embedding dimension
- **Space complexity:** O(N·D) for the in-memory FAISS index
- **Startup optimisation:** Embeddings, index, and model are cached via `@st.cache_resource`
- For larger corpora (>100k chunks), consider `faiss.IndexIVFFlat` for sub-linear search

---

## 🧠 How It Works

```
Article Text
    │
    ▼
Chunking (sentences/paragraphs)
    │
    ▼
Embedding (all-MiniLM-L6-v2)  ──► embeddings.npy
    │
    ▼
FAISS Index (IndexFlatIP)
    │
    ▼
User Query ──► Embed Query ──► FAISS Search ──► Top-K Chunks
                                                       │
                                                       ▼
                                               Build RAG Prompt
                                                       │
                                                       ▼
                                          Llama-3.2-3B-Instruct
                                                       │
                                                       ▼
                                              Cited Answer ✅
```
