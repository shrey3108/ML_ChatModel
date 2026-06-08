"""
llm_client.py
-------------
Wraps the HuggingFace InferenceClient for chat completions with
input validation, error handling, and configurable generation parameters.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
_MAX_QUERY_LENGTH = 1000  # characters — basic input sanitisation


def build_prompt(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """Construct a RAG prompt from the user query and retrieved context chunks.

    Args:
        query:            The user's question (already validated).
        retrieved_chunks: List of chunk dicts from :class:`retriever.Retriever`.

    Returns:
        Formatted prompt string ready to be sent to the LLM.
    """
    context_lines = "\n\n".join(
        f"[{c['rank']}] {c['text']}" for c in retrieved_chunks
    )
    return (
        "You are a helpful AI assistant. Use ONLY the context below to answer "
        "the question concisely. Cite sources using their bracket numbers like [1], [2].\n"
        "If the answer cannot be found in the context, respond with: "
        "\"I don't know based on the provided context.\"\n\n"
        f"Context:\n{context_lines}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


class LLMClient:
    """Thin wrapper around HuggingFace InferenceClient for chat completions.

    Attributes:
        model:  HuggingFace model identifier used for generation.
        client: Underlying :class:`huggingface_hub.InferenceClient`.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        token: str | None = None,
    ) -> None:
        """Initialise the LLM client.

        Args:
            model: HuggingFace model repo ID.
            token: HuggingFace API token. Falls back to ``HUGGINGFACE_TOKEN``
                   environment variable if *None*.

        Raises:
            EnvironmentError: If no token is available.
        """
        resolved_token = token or os.getenv("HUGGINGFACE_TOKEN")
        if not resolved_token:
            raise EnvironmentError(
                "HuggingFace token not found. Set HUGGINGFACE_TOKEN in your .env file."
            )
        self.model = model
        self.client = InferenceClient(model=model, token=resolved_token)
        logger.info("LLMClient initialised with model: %s", model)

    def generate_answer(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.3,
    ) -> str:
        """Send the prompt to the LLM and return the generated text.

        Args:
            prompt:         The fully constructed RAG prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature (lower = more deterministic).

        Returns:
            Generated answer string.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            # Access attribute (not dict key) — correct HuggingFace SDK usage
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc


def sanitise_query(query: str) -> str:
    """Basic input sanitisation for user queries.

    - Strips leading/trailing whitespace.
    - Truncates to ``_MAX_QUERY_LENGTH`` characters.
    - Raises ValueError for empty input.

    Args:
        query: Raw user input string.

    Returns:
        Sanitised query string.

    Raises:
        ValueError: If query is empty after stripping.
    """
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Query cannot be empty.")
    if len(cleaned) > _MAX_QUERY_LENGTH:
        logger.warning(
            "Query truncated from %d to %d characters.", len(cleaned), _MAX_QUERY_LENGTH
        )
        cleaned = cleaned[:_MAX_QUERY_LENGTH]
    return cleaned
