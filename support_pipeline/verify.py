"""Separate, deterministic compliance verifier.

This stage never calls an LLM. It checks the draft against the retrieved policy
chunks, the deterministic gate mode, and a configurable set of prohibited claims.
"""

from __future__ import annotations

import re
from typing import Any

from .index import tokenize
from .ingest import split_sentences
from .io_utils import sha256_text, utc_now_iso, unique_preserving_order
from .retrieve import retrieved_chunk_ids

VERIFIER_VERSION = "deterministic-1.0"

NEGATION_RE = re.compile(
    r"\b(?:not|never|no|cannot|can['\u2019]?t|won['\u2019]?t|don['\u2019]?t|"
    r"isn['\u2019]?t|aren['\u2019]?t|doesn['\u2019]?t|didn['\u2019]?t|couldn['\u2019]?t|"
    r"wouldn['\u2019]?t|shouldn['\u2019]?t|will\s+not|unable|must\s+not|"
    r"do\s+not|without|no\s+longer)\b",
    re.IGNORECASE,
)


def _issue(issue_type: str, text: str, supporting: list[str], rule_id: str | None = None) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "type": issue_type,
        "text": text,
        "supporting_chunk_ids": unique_preserving_order(supporting),
    }
    if rule_id:
        issue["rule_id"] = rule_id
    return issue


def _is_courtesy(sentence: str, config: dict[str, Any]) -> bool:
    for pattern in config.get("verification", {}).get("courtesy_patterns", []):
        if re.match(pattern, sentence, re.IGNORECASE):
            return True
    return False


def _is_safe_phrase(sentence: str, config: dict[str, Any]) -> bool:
    lowered = sentence.lower()
    for phrase in config.get("verification", {}).get("safe_phrases", []):
        if phrase.lower() in lowered:
            return True
    return False


def _is_negated(sentence: str, match_start: int, match_end: int, window: int) -> bool:
    prefix = sentence[max(0, match_start - window) : match_start]
    matched = sentence[match_start:match_end]
    return bool(NEGATION_RE.search(prefix)) or bool(NEGATION_RE.search(matched))


def _supporting_chunks_for_category(
    category: str,
    retrieved_ids: list[str],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    category_terms = {
        "refund": ["refund", "money", "deposit", "payment"],
        "verification": ["verification", "verify", "identity", "kyc", "approval"],
        "account_closure": ["closure", "close", "account"],
        "password": ["password", "reset"],
        "generic": [],
    }.get(category, [])
    if not category_terms:
        return retrieved_ids[:2]

    candidates: list[tuple[int, str]] = []
    for chunk_id in retrieved_ids:
        chunk = corpus_by_id.get(chunk_id)
        if not chunk:
            continue
        text = (chunk.get("title", "") + " " + chunk.get("text", "")).lower()
        term_hits = sum(1 for term in category_terms if term in text)
        prohibition_hits = sum(
            1 for marker in ("must not", "cannot", "not promise", "not guarantee", "depend", "delay", "review")
            if marker in text
        )
        candidates.append((term_hits * 2 + prohibition_hits, chunk_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [chunk_id for score, chunk_id in candidates if score > 0][:2]
    return selected or retrieved_ids[:2]


def _containment(tokens: set[str], chunk: dict[str, Any], config: dict[str, Any]) -> float:
    if not tokens:
        return 0.0
    chunk_tokens = set(tokenize(chunk.get("title", "") + " " + chunk.get("text", ""), config))
    if not chunk_tokens:
        return 0.0
    return len(tokens & chunk_tokens) / len(tokens)


def _check_prohibited(
    reply: str,
    retrieved_ids: list[str],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verification = config.get("verification", {})
    window = int(verification.get("negation_window_chars", 100))
    hedge_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in verification.get("hedge_patterns", [])
    ]
    issues: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for rule in verification.get("prohibited_patterns", []):
        try:
            compiled = re.compile(rule["pattern"], re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid verifier pattern {rule.get('id')}: {exc}") from exc
        for sentence in split_sentences(reply):
            courtesy = _is_courtesy(sentence, config)
            for match in compiled.finditer(sentence):
                if courtesy and rule.get("type") == "unsupported_claim":
                    continue
                if any(pattern.search(sentence) for pattern in hedge_patterns) and rule.get("id") == "absolute_timing":
                    continue
                if _is_negated(sentence, match.start(), match.end(), window):
                    continue
                key = (rule.get("type", "unsupported_claim"), sentence, rule.get("id", ""))
                if key in seen:
                    continue
                seen.add(key)
                supporting = _supporting_chunks_for_category(
                    rule.get("category", "generic"), retrieved_ids, corpus_by_id, config
                )
                finding = _issue(
                    rule.get("type", "unsupported_claim"),
                    sentence,
                    supporting,
                    rule_id=rule.get("id"),
                )
                issues.append(finding)
                findings.append(
                    {
                        "rule_id": rule.get("id"),
                        "sentence": sentence,
                        "matched_text": match.group(0),
                    }
                )
    return issues, findings


def _check_citations(
    used_chunk_ids: list[str],
    retrieved_ids: list[str],
    is_fallback: bool,
    insufficient_policy_support: bool,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not used_chunk_ids:
        if not (is_fallback and insufficient_policy_support):
            issues.append(
                _issue(
                    "policy_mismatch",
                    "Draft does not cite any retrieved policy chunks.",
                    [],
                    rule_id="citation.missing",
                )
            )
        return issues

    invalid = [chunk_id for chunk_id in used_chunk_ids if chunk_id not in set(retrieved_ids)]
    if invalid:
        issues.append(
            _issue(
                "policy_mismatch",
                "Draft cites chunk IDs that were not retrieved: " + ", ".join(invalid),
                [chunk_id for chunk_id in used_chunk_ids if chunk_id in set(retrieved_ids)],
                rule_id="citation.not_retrieved",
            )
        )
    return issues


def _check_grounding(
    reply: str,
    used_chunk_ids: list[str],
    retrieved_ids: list[str],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    verification = config.get("verification", {})
    min_tokens = int(verification.get("min_content_tokens", 2))
    threshold = float(verification.get("min_sentence_support", 0.34))
    support_ids = [chunk_id for chunk_id in used_chunk_ids if chunk_id in set(retrieved_ids)]
    if not support_ids:
        support_ids = retrieved_ids[: min(3, len(retrieved_ids))]
    support_chunks = [
        (chunk_id, corpus_by_id[chunk_id])
        for chunk_id in support_ids
        if chunk_id in corpus_by_id
    ]
    if not support_chunks:
        return []

    all_support_text = " ".join(chunk.get("text", "") for _, chunk in support_chunks)
    support_numbers = set(re.findall(r"\d+", all_support_text))

    issues: list[dict[str, Any]] = []
    for sentence in split_sentences(reply):
        if _is_courtesy(sentence, config) or _is_safe_phrase(sentence, config):
            continue
        tokens = set(tokenize(sentence, config))
        if len(tokens) < min_tokens:
            continue

        scored = [
            (_containment(tokens, chunk, config), chunk_id)
            for chunk_id, chunk in support_chunks
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_chunk_id = scored[0]
        if best_score < threshold:
            issues.append(
                _issue(
                    "unsupported_claim",
                    sentence,
                    [best_chunk_id],
                    rule_id="grounding.sentence_support",
                )
            )
            continue

        reply_numbers = set(re.findall(r"\d+", sentence))
        missing_numbers = sorted(reply_numbers - support_numbers)
        if missing_numbers:
            issues.append(
                _issue(
                    "unsupported_claim",
                    sentence,
                    [best_chunk_id],
                    rule_id="grounding.number_not_in_policy",
                )
            )
    return issues


def _check_mode(
    reply: str,
    draft: dict[str, Any],
    gate: dict[str, Any],
    used_chunk_ids: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    gate_mode = gate.get("policy_mode", "standard")
    draft_mode = draft.get("policy_mode", gate_mode)
    if draft_mode != gate_mode:
        issues.append(
            _issue(
                "policy_mismatch",
                f"Draft policy_mode={draft_mode!r} does not match gate policy_mode={gate_mode!r}.",
                used_chunk_ids,
                rule_id="mode.mismatch",
            )
        )

    requirements = config.get("verification", {}).get("mode_requirements", {}).get(gate_mode, {})
    require_any = requirements.get("require_any", [])
    if require_any and not any(term.lower() in reply.lower() for term in require_any):
        issues.append(
            _issue(
                "policy_mismatch",
                f"Reply does not reflect policy_mode={gate_mode!r}; expected one of: "
                + ", ".join(require_any),
                used_chunk_ids,
                rule_id="mode.required_language",
            )
        )
    return issues


def verify_draft(
    draft: dict[str, Any],
    message: dict[str, Any],
    retrieval: dict[str, Any],
    gate: dict[str, Any],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    attempt: int = 1,
    is_fallback: bool = False,
) -> dict[str, Any]:
    reply = str(draft.get("reply", ""))
    check_reply = reply.replace("\u2019", "'").replace("\u2018", "'")
    used_chunk_ids = unique_preserving_order(
        [str(chunk_id) for chunk_id in draft.get("used_chunk_ids", [])]
    )
    retrieved_ids = retrieved_chunk_ids(retrieval)

    citation_issues = _check_citations(
        used_chunk_ids,
        retrieved_ids,
        is_fallback=is_fallback,
        insufficient_policy_support=bool(draft.get("insufficient_policy_support")),
    )
    prohibited_issues, prohibited_findings = _check_prohibited(
        check_reply, retrieved_ids, corpus_by_id, config
    )
    grounding_issues = _check_grounding(
        check_reply, used_chunk_ids, retrieved_ids, corpus_by_id, config
    )
    mode_issues = _check_mode(check_reply, draft, gate, used_chunk_ids, config)

    issues: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for issue in citation_issues + prohibited_issues + grounding_issues + mode_issues:
        key = (issue["type"], issue["text"])
        if key not in seen:
            seen.add(key)
            issues.append(issue)

    issue_counts: dict[str, int] = {}
    for issue in issues:
        issue_counts[issue["type"]] = issue_counts.get(issue["type"], 0) + 1

    checks = {
        "citation_integrity": not citation_issues,
        "prohibited_promises": not prohibited_issues,
        "grounding": not grounding_issues,
        "mode_compliance": not mode_issues,
    }
    return {
        "message_id": message["message_id"],
        "approved": not issues,
        "issues": issues,
        "attempt": attempt,
        "is_fallback": is_fallback,
        "draft_hash": sha256_text(reply),
        "verifier_version": VERIFIER_VERSION,
        "checks": checks,
        "issue_counts": issue_counts,
        "prohibited_findings": prohibited_findings,
        "retrieved_chunk_ids": retrieved_ids,
        "verified_at": utc_now_iso(),
    }
