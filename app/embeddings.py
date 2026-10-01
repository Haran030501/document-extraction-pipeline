"""Local text embeddings (fastembed / ONNX, CPU). No API key or network needed after the first download."""

from functools import lru_cache

import numpy as np

from app.config import get_settings


@lru_cache
def _model():
    from fastembed import TextEmbedding

    s = get_settings()
    return TextEmbedding(s.embedding_model, cache_dir=s.embedding_cache_dir)


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    return list(_model().embed(texts, batch_size=32))


def embed_query(text: str) -> np.ndarray:
    # bge models expect an instruction prefix on queries; fastembed's query_embed adds it.
    return next(iter(_model().query_embed([text])))
