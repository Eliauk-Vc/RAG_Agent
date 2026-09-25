"""Bound model inputs without silently summarizing user facts."""

import json


def packed(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def clip_text(text: str, limit: int) -> str:
    return text.encode("utf-8")[:max(0, limit)].decode("utf-8", errors="ignore")


def recent_history(history: list[dict], turns: int, budget: int) -> list[dict]:
    """Keep the newest complete turns that fit; full originals stay in SQLite."""
    selected = []
    for end in range(len(history), max(0, len(history) - turns * 2), -2):
        pair = history[max(0, end - 2):end]
        candidate = pair + selected
        if len(packed(candidate).encode("utf-8")) > budget:
            break
        selected = candidate
    return selected


def merge_evidence(batches: list[list[dict]], budget: int) -> dict[str, dict]:
    """Share the evidence budget across searches so the first query cannot monopolize it."""
    selected = {}
    for rank in range(max((len(batch) for batch in batches), default=0)):
        for batch in reversed(batches):
            if rank >= len(batch):
                continue
            chunk = batch[rank]
            identifier = chunk["chunk_id"]
            if identifier in selected:
                continue
            candidate = list(selected.values()) + [chunk]
            if len(packed(candidate).encode("utf-8")) <= budget:
                selected[identifier] = chunk
    return selected


def rrf_evidence(batches: list[list[dict]], budget: int, k: int = 60) -> dict[str, dict]:
    """Fuse ordered subtask evidence lists by reciprocal rank, not factual confidence.

    Each chunk votes once per list. Ranks start at 1 after within-list deduplication.
    Equal scores retain first appearance order. Input records are never modified.
    """
    if k <= 0 or budget <= 0:
        raise ValueError('RRF k and evidence budget must be positive')
    candidates = {}
    for batch in batches:
        seen = set()
        for chunk in batch:
            identifier = chunk['chunk_id']
            if identifier in seen:
                continue
            seen.add(identifier)
            if identifier not in candidates:
                candidates[identifier] = {**chunk, 'rrf_score': 0.0, 'rrf_support': 0}
            record = candidates[identifier]
            record['rrf_score'] += 1.0 / (k + len(seen))
            record['rrf_support'] += 1
    selected = {}
    for chunk in sorted(candidates.values(), key=lambda row: -row['rrf_score']):
        candidate = list(selected.values()) + [chunk]
        if len(packed(candidate).encode('utf-8')) <= budget:
            selected[chunk['chunk_id']] = chunk
    return selected


def bounded_payload(system: str, payload: dict, limit: int) -> str:
    """Drop older context/evidence as needed, never truncate the current question."""
    payload = dict(payload)
    payload["history"] = list(payload.get("history", []))
    payload["evidence"] = list(payload.get("evidence", []))
    payload["observations"] = list(payload.get("observations", []))
    while len((system + packed(payload)).encode("utf-8")) > limit:
        if payload["history"]:
            payload["history"] = payload["history"][2:]
        elif len(payload["observations"]) > 1:
            payload["observations"].pop(0)
        elif payload["observations"] and any(
            key in payload["observations"][0] for key in ("nodes", "edges", "entities")
        ):
            payload["observations"][0] = {
                key: value for key, value in payload["observations"][0].items()
                if key not in {"nodes", "edges", "entities"}
            }
        elif len(payload["evidence"]) > 1:
            payload["evidence"].pop()
        elif payload["evidence"] and len(payload["evidence"][0]["content"].encode("utf-8")) > 500:
            first = dict(payload["evidence"][0])
            first["content"] = clip_text(first["content"], len(first["content"].encode("utf-8")) // 2)
            first["truncated"] = True
            payload["evidence"][0] = first
        elif payload["observations"]:
            payload["observations"] = []
        else:
            raise ValueError("Question and required state exceed the model input budget")
    return packed(payload)
