"""Ephemeral, per-search embeddings shared by relevance ranking and MMR."""
import asyncio
from contextvars import ContextVar
import numpy as np

SEARCH_EMBEDDINGS = ContextVar('search_embeddings', default=None)


class SearchEmbeddings:
    def __init__(self, model_identity=None):
        self.model_identity = model_identity
        self.vectors = {}
        self.originals = {}
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0


async def embed_once(embedding, texts, context):
    cache = SEARCH_EMBEDDINGS.get()
    if cache is None:
        return await embedding(texts, context=context)
    # LightRAG wraps EmbeddingFunc via dataclasses.replace. Both the reranker and
    # MMR use the search's configured model, even though their wrapper IDs differ.
    async with cache.lock:
        identity = cache.model_identity if cache.model_identity is not None else id(embedding)
        keys = [(identity, context, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in cache.vectors))
        cache.hits += len(keys) - len(missing)
        if missing:
            vectors = np.asarray(await embedding([key[2] for key in missing], context=context), dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(missing) or not np.isfinite(vectors).all():
                raise ValueError('Invalid embeddings for reuse')
            cache.vectors.update((key, vector.copy()) for key, vector in zip(missing, vectors))
            cache.misses += len(missing)
        return np.asarray([cache.vectors[key] for key in keys]).copy()
