"""Top-k BM25 retrieval over the local policy index."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .index import index_fingerprint, tokenize
from .io_utils import short_hash, utc_now_iso


def score_chunk(
    query_terms: Counter[str],
    chunk: dict[str, Any],
    index: dict[str, Any],
) -> tuple[float, list[str]]:
    k1 = float(index.get("k1", 1.5))
    b = float(index.get("b", 0.75))
    avgdl = float(index.get("avgdl", 0.0)) or 1.0
    doc_length = float(chunk.get("length", 0.0))
    idf = index.get("idf", {})
    term_freq = chunk.get("term_freq", {})

    score = 0.0
    matched: list[str] = []
    for term, query_count in query_terms.items():
        frequency = float(term_freq.get(term, 0))
        if frequency <= 0:
            continue
        matched.append(term)
        term_idf = float(idf.get(term, 0.0))
        denominator = frequency + k1 * (1.0 - b + b * (doc_length / avgdl))
        score += query_count * term_idf * (frequency * (k1 + 1.0) / denominator)
    return score, sorted(matched)


def retrieve(message: dict[str, Any], index: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    message_id = message["message_id"]
    query = message["customer_message"]
    query_tokens = tokenize(query, config)
    query_terms = Counter(query_tokens)
    retrieval_config = config.get("retrieval", {})
    top_k = int(retrieval_config.get("top_k", 3))
    min_score = float(retrieval_config.get("min_relevance_score", 0.0))

    scored: list[tuple[float, list[str], dict[str, Any]]] = []
    for chunk in index.get("chunks", []):
        score, matched = score_chunk(query_terms, chunk, index)
        scored.append((score, matched, chunk))

    scored.sort(key=lambda item: (-item[0], item[2].get("chunk_id", "")))
    selected = scored[:top_k]
    top_score = selected[0][0] if selected else 0.0

    results: list[dict[str, Any]] = []
    for rank, (score, matched, chunk) in enumerate(selected, start=1):
        normalized = (score / top_score) if top_score > 0 else 0.0
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
                "content_hash": chunk["content_hash"],
                "score": round(score, 6),
                "score_normalized": round(normalized, 6),
                "matched_terms": matched,
                "text": chunk["text"],
            }
        )

    weak_support = (not results) or (top_score < min_score) or (not results[0]["matched_terms"])
    fingerprint = index_fingerprint(index)
    return {
        "retrieval_id": "RET-" + message_id + "-" + short_hash(query + fingerprint, 8),
        "message_id": message_id,
        "query": query,
        "query_tokens": query_tokens,
        "top_k": top_k,
        "returned_count": len(results),
        "weak_support": weak_support,
        "top_score": round(top_score, 6),
        "min_relevance_score": min_score,
        "index_fingerprint": fingerprint,
        "results": results,
        "created_at": utc_now_iso(),
    }


def retrieve_all(
    messages: list[dict[str, Any]], index: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    return [retrieve(message, index, config) for message in messages]


def retrieved_chunk_ids(retrieval: dict[str, Any]) -> list[str]:
    return [result["chunk_id"] for result in retrieval.get("results", [])]
