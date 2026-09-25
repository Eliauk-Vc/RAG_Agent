"""Greedy maximal marginal relevance over bounded semantic candidates."""
import numpy as np


def mmr_indices(query_vector, document_vectors, top_k, relevance_weight=0.7):
    """Choose first by query cosine, then lambda*relevance-(1-lambda)*redundancy.

    Ties preserve input order. Invalid embeddings must trigger caller fallback.
    """
    if top_k < 1 or not 0 <= relevance_weight <= 1:
        raise ValueError('Invalid MMR parameters')
    query = np.asarray(query_vector, dtype=float)
    docs = np.asarray(document_vectors, dtype=float)
    if query.ndim != 1 or docs.ndim != 2 or docs.shape[1] != len(query):
        raise ValueError('Invalid embedding shape')
    if not np.isfinite(query).all() or not np.isfinite(docs).all():
        raise ValueError('Nonfinite embedding')
    if not len(docs):
        return []
    norms = np.linalg.norm(docs, axis=1)
    qnorm = np.linalg.norm(query)
    if qnorm <= 0 or (norms <= 0).any():
        raise ValueError('Zero embedding')
    docs = docs / norms[:, None]
    relevance = docs @ (query / qnorm)
    similarity = docs @ docs.T
    selected = [int(np.argmax(relevance))]
    while len(selected) < min(top_k, len(docs)):
        scores = relevance_weight * relevance - (1 - relevance_weight) * similarity[:, selected].max(axis=1)
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    return selected
