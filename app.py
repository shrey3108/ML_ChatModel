"""
app.py
------
Streamlit entry-point for the RAG-based ML Q&A Bot.

Architecture:
    app.py          ← UI layer (Streamlit)
        └─ retriever.py     ← FAISS semantic search
        └─ embedder.py      ← Sentence-Transformer encoding
        └─ llm_client.py    ← HuggingFace LLM + prompt builder

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from embedder import embed_query, load_embedder
from llm_client import LLMClient, build_prompt, sanitise_query
from retriever import Retriever

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached resource loading (runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    """Load SentenceTransformer model once and cache it."""
    return load_embedder()


@st.cache_resource(show_spinner="Building FAISS index…")
def get_retriever():
    """Load precomputed embeddings/chunks and build FAISS index once."""
    return Retriever(embeddings_path="embeddings.npy", chunks_path="chunks.json")


@st.cache_resource(show_spinner="Connecting to LLM…")
def get_llm_client():
    """Initialise HuggingFace InferenceClient once."""
    return LLMClient()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Q&A Bot — ML Knowledge Base",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 RAG Demo — ML Knowledge Base")
st.markdown(
    "Ask any question about the ingested ML articles. "
    "The system retrieves relevant passages and generates a cited answer using **Llama-3.2**."
)

# Sidebar — configuration
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=3)
    max_tokens = st.slider("Max response tokens", min_value=100, max_value=500, value=300)
    st.markdown("---")
    st.caption("Model: `meta-llama/Llama-3.2-3B-Instruct`")
    st.caption("Embeddings: `all-MiniLM-L6-v2`")

# Load resources
try:
    embedder = get_embedder()
    retriever = get_retriever()
    llm = get_llm_client()
except EnvironmentError as env_err:
    st.error(f"⚠️ Configuration error: {env_err}")
    st.info("Create a `.env` file with `HUGGINGFACE_TOKEN=your_token_here`.")
    st.stop()
except FileNotFoundError as file_err:
    st.error(f"⚠️ Data file missing: {file_err}")
    st.stop()

# Query input
query_raw = st.text_input(
    "Ask a question about the ML articles:",
    placeholder="e.g. How does FAISS improve retrieval speed?",
)

if st.button("🔎 Search & Answer", disabled=not query_raw):
    # 1. Sanitise input
    try:
        query = sanitise_query(query_raw)
    except ValueError as ve:
        st.warning(str(ve))
        st.stop()

    # 2. Embed query
    with st.spinner("Embedding query…"):
        try:
            q_emb = embed_query(embedder, query)
        except Exception as emb_err:
            st.error(f"Embedding failed: {emb_err}")
            st.stop()

    # 3. Retrieve chunks
    with st.spinner(f"Retrieving top-{top_k} chunks…"):
        try:
            retrieved = retriever.retrieve(q_emb, k=top_k)
        except Exception as ret_err:
            st.error(f"Retrieval failed: {ret_err}")
            st.stop()

    if not retrieved:
        st.warning("No relevant chunks found. Try a different question.")
        st.stop()

    # 4. Build prompt & generate answer
    prompt = build_prompt(query, retrieved)
    with st.spinner("Generating answer with Llama-3.2…"):
        try:
            answer = llm.generate_answer(prompt, max_new_tokens=max_tokens)
        except RuntimeError as llm_err:
            st.error(f"LLM call failed: {llm_err}")
            st.stop()

    # 5. Display results
    st.markdown("### 📘 Answer")
    st.write(answer)

    with st.expander("📄 Retrieved Chunks (Context Used)", expanded=False):
        for chunk in retrieved:
            st.markdown(
                f"**[{chunk['rank']}]** — Score: `{chunk['score']}`\n\n{chunk['text']}"
            )
            st.divider()
