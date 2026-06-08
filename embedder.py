"""
embedder.py
-----------
Handles loading the sentence-transformer embedding model and encoding queries.
Model is cached so it is only loaded once per Streamlit session.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder(model_name: str = _MODEL_NAME) -> SentenceTransformer:
    """Load and return the SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name, device="cpu")


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Encode a single query string into a normalised embedding vector.

    Args:
        model: A loaded SentenceTransformer instance.
        query: The user question to embed.

    Returns:
        1-D float32 numpy array (L2-normalised).

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    embedding: np.ndarray = model.encode([query], convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)
