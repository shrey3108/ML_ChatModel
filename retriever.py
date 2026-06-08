"""
retriever.py
------------
Handles loading precomputed embeddings and chunk data, building a FAISS index,
and retrieving the top-k most relevant chunks for a given query embedding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """FAISS-backed semantic retriever for RAG pipelines.

    The retriever loads precomputed embeddings (.npy) and corresponding text
    chunks (.json), normalises the embeddings, and builds an inner-product
    FAISS index (equivalent to cosine similarity after L2-normalisation).

    Attributes:
        chunks: List of chunk dicts loaded from disk.
        index:  Compiled FAISS index ready for search.
        dim:    Embedding dimension.
    """

    def __init__(
        self,
        embeddings_path: str | Path = "embeddings.npy",
        chunks_path: str | Path = "chunks.json",
    ) -> None:
        """Initialise the retriever by loading embeddings and chunks from disk.

        Args:
            embeddings_path: Path to the precomputed embeddings numpy file.
            chunks_path:     Path to the JSON file containing text chunks.

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError:        If embeddings and chunks have mismatched lengths.
        """
        embeddings_path = Path(embeddings_path)
        chunks_path = Path(chunks_path)

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        # Load chunks
        with chunks_path.open("r", encoding="utf-8") as fh:
            self.chunks: list[dict[str, Any]] = json.load(fh)

        # Load and normalise embeddings
        raw: np.ndarray = np.load(str(embeddings_path)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        embeddings_norm = raw / norms

        if len(embeddings_norm) != len(self.chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings_norm)} embeddings vs "
                f"{len(self.chunks)} chunks."
            )

        # Build FAISS index (Inner Product = cosine sim on normalised vectors)
        self.dim: int = embeddings_norm.shape[1]
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings_norm)
        logger.info(
            "FAISS index built with %d vectors (dim=%d).", len(self.chunks), self.dim
        )

    def retrieve(self, query_embedding: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
        """Return the top-k most relevant chunks for the query embedding.

        Args:
            query_embedding: 1-D float32 normalised embedding vector.
            k:               Number of top chunks to retrieve.

        Returns:
            List of dicts with keys ``chunk_id``, ``score``, and ``text``.

        Raises:
            ValueError: If k is not a positive integer.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")

        query_2d = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_2d, k)

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self.chunks):
                logger.warning("FAISS returned out-of-range index %d; skipping.", idx)
                continue
            results.append(
                {
                    "rank": rank + 1,
                    "chunk_id": int(idx),
                    "score": round(float(score), 4),
                    "text": self.chunks[idx]["text"],
                }
            )
        return results
