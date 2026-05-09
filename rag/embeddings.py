"""Hugging Face cloud embedding client for the RAG pipeline.

All embeddings are generated via the Hugging Face Inference API,
so no local ML model (sentence-transformers / torch) is loaded.
This keeps the backend lightweight and Render-free-tier friendly.
"""

from __future__ import annotations

import logging
import os

from huggingface_hub import InferenceClient

from rag.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def _get_hf_token() -> str:
    """Read the Hugging Face API token from the environment."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. "
            "Add it to your .env file or set it as an environment variable."
        )
    return token


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts via the HF Inference API.

    Args:
        texts: The strings to embed.

    Returns:
        A list of embedding vectors (one per input text).

    Raises:
        RuntimeError: If HF_TOKEN is missing or the API call fails.
    """
    token = _get_hf_token()
    client = InferenceClient(token=token)

    try:
        result = client.feature_extraction(
            text=texts,
            model=EMBEDDING_MODEL_NAME,
        )
    except Exception as error:
        raise RuntimeError(
            f"Hugging Face Inference API call failed: {error}"
        ) from error

    # The API returns nested lists; ensure we return list[list[float]].
    embeddings: list[list[float]] = [
        [float(value) for value in vector] for vector in result
    ]

    logger.debug(
        "Generated %d embeddings via HF cloud (%s)",
        len(embeddings),
        EMBEDDING_MODEL_NAME,
    )

    return embeddings
