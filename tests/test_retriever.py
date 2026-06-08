"""
tests/test_retriever.py
-----------------------
Unit tests for the Retriever and embedder modules.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from retriever import Retriever
from embedder import embed_query, load_embedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_data(tmp_path: Path):
    """Create temporary chunks.json and embeddings.npy for testing."""
    chunks = [
        {"text": "FAISS is a library for efficient similarity search."},
        {"text": "Streamlit allows you to build web apps in Python easily."},
        {"text": "Sentence transformers produce dense embedding vectors."},
    ]
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    # Random normalised embeddings (dim=64)
    np.random.seed(42)
    embeddings = np.random.rand(3, 64).astype(np.float32)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(str(embeddings_path), embeddings)

    return str(embeddings_path), str(chunks_path)


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

class TestRetriever:

    def test_loads_successfully(self, sample_data):
        emb_path, chunk_path = sample_data
        retriever = Retriever(emb_path, chunk_path)
        assert retriever.index.ntotal == 3
        assert len(retriever.chunks) == 3

    def test_retrieve_returns_correct_count(self, sample_data):
        emb_path, chunk_path = sample_data
        retriever = Retriever(emb_path, chunk_path)
        query_emb = np.random.rand(64).astype(np.float32)
        results = retriever.retrieve(query_emb, k=2)
        assert len(results) == 2

    def test_retrieve_result_has_required_keys(self, sample_data):
        emb_path, chunk_path = sample_data
        retriever = Retriever(emb_path, chunk_path)
        query_emb = np.random.rand(64).astype(np.float32)
        results = retriever.retrieve(query_emb, k=1)
        assert len(results) == 1
        assert {"rank", "chunk_id", "score", "text"}.issubset(results[0].keys())

    def test_retrieve_raises_on_invalid_k(self, sample_data):
        emb_path, chunk_path = sample_data
        retriever = Retriever(emb_path, chunk_path)
        query_emb = np.random.rand(64).astype(np.float32)
        with pytest.raises(ValueError, match="positive integer"):
            retriever.retrieve(query_emb, k=0)

    def test_missing_embeddings_file(self, tmp_path):
        chunks = [{"text": "hello"}]
        chunk_path = tmp_path / "chunks.json"
        chunk_path.write_text(json.dumps(chunks))
        with pytest.raises(FileNotFoundError):
            Retriever(str(tmp_path / "missing.npy"), str(chunk_path))

    def test_missing_chunks_file(self, tmp_path):
        emb = np.random.rand(3, 64).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(str(emb_path), emb)
        with pytest.raises(FileNotFoundError):
            Retriever(str(emb_path), str(tmp_path / "missing.json"))

    def test_mismatch_raises_error(self, tmp_path):
        chunks = [{"text": "a"}, {"text": "b"}]
        chunk_path = tmp_path / "chunks.json"
        chunk_path.write_text(json.dumps(chunks))
        # 3 embeddings but only 2 chunks
        emb = np.random.rand(3, 64).astype(np.float32)
        emb_path = tmp_path / "embeddings.npy"
        np.save(str(emb_path), emb)
        with pytest.raises(ValueError, match="Mismatch"):
            Retriever(str(emb_path), str(chunk_path))


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestEmbedder:

    def test_embed_query_returns_float32(self):
        model = load_embedder()
        result = embed_query(model, "What is FAISS?")
        assert result.dtype == np.float32

    def test_embed_query_is_normalised(self):
        model = load_embedder()
        result = embed_query(model, "How do embeddings work?")
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_query_raises_on_empty(self):
        model = load_embedder()
        with pytest.raises(ValueError, match="non-empty"):
            embed_query(model, "")

    def test_embed_query_raises_on_whitespace(self):
        model = load_embedder()
        with pytest.raises(ValueError, match="non-empty"):
            embed_query(model, "   ")
