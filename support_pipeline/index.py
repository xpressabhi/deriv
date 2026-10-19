"""File-based BM25 index with content-hash aware token caching."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import retrieval_config_hash
from .ingest import corpus_hash
from .io_utils import ensure_dir, read_json, sha256_json, utc_now_iso, write_json


WORD_RE = re.compile(r"[a-z0-9]+")
INDEX_VERSION = "1.0"
INDEX_FILE = "bm25_index.json"
CACHE_FILE = "chunk_token_cache.json"
MANIFEST_FILE = "index_manifest.json"


def _normalize_text(text: str) -> str:
    return (
        (text or "")
        .lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _plural_stem(token: str) -> str:
    exceptions = {
        "status", "business", "analysis", "process", "access", "address",
        "success", "class", "case", "base", "release", "response", "purpose",
    }
    if token in exceptions or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "zes", "ces")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, config: dict[str, Any]) -> list[str]:
    retrieval = config.get("retrieval", {})
    stopwords = {word.lower() for word in retrieval.get("stopwords", [])}
    phrase_synonyms = retrieval.get("phrase_synonyms", {})
    token_synonyms = retrieval.get("token_synonyms", {})

    normalized = _normalize_text(text)
    for phrase in sorted(phrase_synonyms, key=len, reverse=True):
        replacement = str(phrase_synonyms[phrase])
        normalized = re.sub(r"\b" + re.escape(phrase.lower()) + r"\b", replacement.lower(), normalized)

    raw_tokens = [token for token in WORD_RE.findall(normalized) if token not in stopwords]
    tokens: list[str] = []
    for token in raw_tokens:
        stemmed = _plural_stem(token)
        if stemmed and stemmed not in stopwords:
            tokens.append(stemmed)
        for synonym in token_synonyms.get(stemmed, []):
            candidate = _plural_stem(str(synonym).lower())
            if candidate and candidate not in stopwords:
                tokens.append(candidate)
    return tokens


def _tokenize_chunk(chunk: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    tokens = tokenize(chunk["title"] + " " + chunk["text"], config)
    term_freq = Counter(tokens)
    return {
        "length": len(tokens),
        "term_freq": dict(sorted(term_freq.items())),
    }


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
        entries = payload.get("entries", {})
        return entries if isinstance(entries, dict) else {}
    except (ValueError, OSError):
        return {}


def _diff_chunks(
    chunks: list[dict[str, Any]], previous_hashes: dict[str, str]
) -> tuple[list[str], list[str], list[str], int]:
    current_hashes = {chunk["chunk_id"]: chunk["content_hash"] for chunk in chunks}
    added = sorted(set(current_hashes) - set(previous_hashes))
    removed = sorted(set(previous_hashes) - set(current_hashes))
    modified = sorted(
        chunk_id
        for chunk_id in set(current_hashes) & set(previous_hashes)
        if current_hashes[chunk_id] != previous_hashes[chunk_id]
    )
    unchanged = len(set(current_hashes) & set(previous_hashes)) - len(modified)
    return added, removed, modified, unchanged


def build_index(
    chunks: list[dict[str, Any]],
    config: dict[str, Any],
    index_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = ensure_dir(index_dir)
    index_path = directory / INDEX_FILE
    cache_path = directory / CACHE_FILE
    manifest_path = directory / MANIFEST_FILE

    current_corpus_hash = corpus_hash(chunks)
    current_config_hash = retrieval_config_hash(config)
    current_hashes = {chunk["chunk_id"]: chunk["content_hash"] for chunk in chunks}

    previous_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous_manifest = read_json(manifest_path)
        except (ValueError, OSError):
            previous_manifest = {}

    previous_hashes = previous_manifest.get("chunk_hashes", {}) if isinstance(previous_manifest, dict) else {}
    added, removed, modified, unchanged = _diff_chunks(chunks, previous_hashes)

    can_reuse = (
        index_path.exists()
        and previous_manifest.get("corpus_hash") == current_corpus_hash
        and previous_manifest.get("retrieval_config_hash") == current_config_hash
    )

    if can_reuse:
        index = read_json(index_path)
        report = {
            "corpus_hash": current_corpus_hash,
            "retrieval_config_hash": current_config_hash,
            "changed": False,
            "action": "reused",
            "added_chunk_ids": [],
            "removed_chunk_ids": [],
            "modified_chunk_ids": [],
            "unchanged_chunk_count": len(chunks),
            "token_cache_hits": 0,
            "token_cache_misses": 0,
            "index_file": str(index_path),
            "generated_at": utc_now_iso(),
        }
        write_json(directory / "chunk_change_report.json", report)
        return index, report

    reason = "rebuilt"
    if previous_manifest and previous_manifest.get("corpus_hash") == current_corpus_hash:
        reason = "rebuilt_config_changed"
    elif previous_manifest:
        reason = "rebuilt_corpus_changed"

    cache = _load_cache(cache_path)
    cache_hits = 0
    cache_misses = 0
    new_cache: dict[str, Any] = {}
    index_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        content_hash = chunk["content_hash"]
        cached = cache.get(content_hash)
        if isinstance(cached, dict) and isinstance(cached.get("term_freq"), dict):
            token_data = {
                "length": int(cached.get("length", 0)),
                "term_freq": cached["term_freq"],
            }
            cache_hits += 1
        else:
            token_data = _tokenize_chunk(chunk, config)
            cache_misses += 1
        new_cache[content_hash] = token_data
        index_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "content_hash": content_hash,
                "length": token_data["length"],
                "term_freq": token_data["term_freq"],
            }
        )

    term_df: Counter[str] = Counter()
    for chunk in index_chunks:
        for term in chunk["term_freq"]:
            term_df[term] += 1

    chunk_count = len(index_chunks)
    idf = {
        term: math.log(1.0 + (chunk_count - df + 0.5) / (df + 0.5))
        for term, df in sorted(term_df.items())
    }
    total_length = sum(chunk["length"] for chunk in index_chunks)
    avgdl = total_length / chunk_count if chunk_count else 0.0
    retrieval = config.get("retrieval", {})

    index = {
        "version": INDEX_VERSION,
        "corpus_hash": current_corpus_hash,
        "retrieval_config_hash": current_config_hash,
        "built_at": utc_now_iso(),
        "chunk_count": chunk_count,
        "avgdl": avgdl,
        "k1": float(retrieval.get("bm25_k1", 1.5)),
        "b": float(retrieval.get("bm25_b", 0.75)),
        "df": dict(sorted(term_df.items())),
        "idf": idf,
        "chunks": index_chunks,
    }
    manifest = {
        "version": INDEX_VERSION,
        "corpus_hash": current_corpus_hash,
        "retrieval_config_hash": current_config_hash,
        "chunk_count": chunk_count,
        "chunk_hashes": current_hashes,
        "built_at": index["built_at"],
        "index_file": INDEX_FILE,
        "token_cache_file": CACHE_FILE,
    }
    report = {
        "corpus_hash": current_corpus_hash,
        "retrieval_config_hash": current_config_hash,
        "changed": True,
        "action": reason,
        "added_chunk_ids": added,
        "removed_chunk_ids": removed,
        "modified_chunk_ids": modified,
        "unchanged_chunk_count": unchanged,
        "token_cache_hits": cache_hits,
        "token_cache_misses": cache_misses,
        "index_file": str(index_path),
        "generated_at": utc_now_iso(),
    }

    write_json(index_path, index)
    write_json(cache_path, {"version": INDEX_VERSION, "entries": new_cache})
    write_json(manifest_path, manifest)
    write_json(directory / "chunk_change_report.json", report)
    return index, report


def index_fingerprint(index: dict[str, Any]) -> str:
    return sha256_json(
        {
            "corpus_hash": index.get("corpus_hash"),
            "retrieval_config_hash": index.get("retrieval_config_hash"),
        }
    )[:12]
