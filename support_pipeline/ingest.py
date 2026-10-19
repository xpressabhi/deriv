"""Policy ingestion and sentence-aware chunking."""

from __future__ import annotations

import re
from typing import Any

from .io_utils import normalize_whitespace, sha256_json, sha256_text


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _documents_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        documents = payload.get("documents")
        if documents is None:
            raise ValueError("policy_docs.json must contain a 'documents' array")
    elif isinstance(payload, list):
        documents = payload
    else:
        raise ValueError("policy_docs.json must be an object with 'documents' or a list")

    if not isinstance(documents, list) or not documents:
        raise ValueError("policy_docs.json contains no documents")

    normalized: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"Policy document at index {index} is not an object")
        doc_id = str(document.get("doc_id") or f"DOC-{index + 1:03d}").strip()
        title = str(document.get("title") or f"Policy {doc_id}").strip()
        text = str(document.get("text") or document.get("content") or "").strip()
        if not text:
            raise ValueError(f"Policy document {doc_id} has no text")
        normalized.append({"doc_id": doc_id, "title": title, "text": text})
    return normalized


def _chunk_text(sentences: list[str], max_chars: int, overlap_sentences: int) -> list[str]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    raw_chunks: list[str] = []
    start = 0
    total = len(sentences)
    while start < total:
        end = start
        current_length = 0
        while end < total:
            sentence = sentences[end]
            addition = len(sentence) + (1 if current_length else 0)
            if end > start and current_length + addition > max_chars:
                break
            current_length += addition
            end += 1
        if end == start:
            end = start + 1
        raw_chunks.append(" ".join(sentences[start:end]))
        if end >= total:
            break
        next_start = end - overlap_sentences
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return raw_chunks


def chunk_document(document: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    chunking = config.get("chunking", {})
    max_chars = int(chunking.get("max_chars", 420))
    overlap = int(chunking.get("overlap_sentences", 1))
    min_chars = int(chunking.get("min_chunk_chars", 40))

    sentences = split_sentences(document["text"])
    raw_chunks = _chunk_text(sentences, max_chars=max_chars, overlap_sentences=overlap)

    # Merge a very short trailing chunk into its predecessor when it fits.
    merged: list[str] = []
    for text in raw_chunks:
        if merged and len(text) < min_chars and len(merged[-1]) + 1 + len(text) <= max_chars * 1.5:
            merged[-1] = merged[-1] + " " + text
        else:
            merged.append(text)

    chunks: list[dict[str, Any]] = []
    for chunk_index, text in enumerate(merged):
        normalized_text = normalize_whitespace(text)
        chunks.append(
            {
                "chunk_id": f"{document['doc_id']}-c{chunk_index:02d}",
                "doc_id": document["doc_id"],
                "title": document["title"],
                "chunk_index": chunk_index,
                "text": normalized_text,
                "content_hash": sha256_text(normalized_text),
            }
        )
    return chunks


def ingest_documents(payload: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for document in _documents_from_payload(payload):
        for chunk in chunk_document(document, config):
            if chunk["chunk_id"] in seen_ids:
                raise ValueError(f"Duplicate chunk_id generated: {chunk['chunk_id']}")
            seen_ids.add(chunk["chunk_id"])
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No policy chunks were produced")
    return chunks


def corpus_hash(chunks: list[dict[str, Any]]) -> str:
    return sha256_json([{"chunk_id": c["chunk_id"], "content_hash": c["content_hash"]} for c in chunks])
