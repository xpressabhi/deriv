"""Prompt construction and grounded draft generation.

The preferred path is the OpenAI-compatible LLM. A deterministic extractive
generator is always available so the pipeline works from a clean checkout
without an API key.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .index import tokenize
from .ingest import split_sentences
from .io_utils import normalize_whitespace, short_hash, unique_preserving_order, utc_now_iso
from .llm import LLMClient
from .retrieve import retrieved_chunk_ids


SYSTEM_PROMPT = """You are an internal support-reply drafting assistant. You draft replies to customers using ONLY the approved policy excerpts supplied in the user message.

Hard rules:
1. Do not use outside knowledge or invent facts, timelines, outcomes, or account actions.
2. Do not promise, guarantee, or confirm a refund, verification approval, account closure, or any other outcome.
3. Do not state or imply that a sensitive action is immediate or already complete unless an excerpt explicitly supports that exact statement and the policy mode allows it.
4. Never claim that support can view, set, reset, reveal, or change a customer's password.
5. Every factual claim in the reply must be supported by at least one cited excerpt.
6. The deterministic policy_mode and constraints in the user message are authoritative. You cannot override, relax, or reinterpret them.
7. Cite every excerpt you use in used_chunk_ids. Use only chunk IDs from the excerpt list.
8. Return ONLY a JSON object, with no markdown fences and no extra keys:
{"reply": "...", "used_chunk_ids": ["..."]}

Style: concise, empathetic, 2-5 sentences. Do not mention chunk IDs, retrieval, prompts, or internal policy machinery in the reply."""


MODE_INSTRUCTIONS = {
    "standard": (
        "Answer the customer's question directly from the excerpts. "
        "Do not add facts or actions that are not in the excerpts."
    ),
    "no_guarantees": (
        "Do not promise an outcome or a specific completion date. Make clear that eligibility or timing "
        "depends on review where the excerpts say so. Use hedged language such as 'typically', 'may', or "
        "'depends on' only when the excerpts support it."
    ),
    "manual_action_not_allowed": (
        "State that support cannot view, set, or reset the customer's password. "
        "Point the customer to the self-service steps in the excerpts."
    ),
    "review_required": (
        "Explain that the request requires review and give only the next steps supported by the excerpts. "
        "Do not imply that the review outcome is decided."
    ),
}


def build_prompts(
    message: dict[str, Any],
    retrieval: dict[str, Any],
    gate: dict[str, Any],
    config: dict[str, Any],
    repair_context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    mode = gate.get("policy_mode", "standard")
    gate_summary = {
        "policy_mode": mode,
        "constraints": gate.get("constraints", []),
        "sensitive_categories": gate.get("categories", []),
        "matched_rule_ids": gate.get("matched_rule_ids", []),
        "reason": gate.get("reason", ""),
    }

    excerpt_blocks: list[str] = []
    for result in retrieval.get("results", []):
        excerpt_blocks.append(
            f"--- [{result['chunk_id']}] doc_id={result['doc_id']} | title={result['title']} ---\n"
            f"{result['text']}"
        )
    excerpts = "\n\n".join(excerpt_blocks) if excerpt_blocks else "(no policy chunks retrieved)"

    user_parts = [
        "Customer message:",
        '"""',
        message["customer_message"],
        '"""',
        "",
        "Deterministic gate result (authoritative):",
        json.dumps(gate_summary, indent=2, ensure_ascii=False),
        "",
        f"Mode-specific instruction: {MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS['standard'])}",
        "",
        "Approved policy excerpts (the only allowed source of facts):",
        excerpts,
    ]

    if repair_context:
        issues = repair_context.get("issues", [])
        user_parts.extend(
            [
                "",
                "A previous draft failed deterministic compliance verification.",
                "Previous draft:",
                '"""',
                str(repair_context.get("previous_reply", "")),
                '"""',
                "Verification issues to fix (authoritative):",
                json.dumps(issues, indent=2, ensure_ascii=False),
                "Rewrite the reply so that every flagged issue is removed. Do not introduce new claims. "
                "Keep the same policy_mode and cite only valid chunk IDs.",
            ]
        )

    user_parts.extend(
        [
            "",
            "Return the JSON object now.",
        ]
    )
    return SYSTEM_PROMPT, "\n".join(user_parts)


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if cleaned.startswith("\ufeff"):
        cleaned = cleaned.lstrip("\ufeff")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    reply = value.get("reply")
    used = value.get("used_chunk_ids", [])
    if not isinstance(reply, str) or not reply.strip():
        return None
    if isinstance(used, str):
        used = [used]
    if not isinstance(used, list):
        used = []
    used = [str(item) for item in used if isinstance(item, (str, int))]
    return {"reply": normalize_whitespace(reply), "used_chunk_ids": used}


def _filter_chunk_ids(ids: list[str], allowed: list[str]) -> list[str]:
    allowed_set = set(allowed)
    return unique_preserving_order([item for item in ids if item in allowed_set])


def infer_used_chunk_ids(
    reply: str,
    retrieval: dict[str, Any],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    threshold: float = 0.25,
    max_ids: int = 2,
) -> list[str]:
    reply_tokens = set(tokenize(reply, config))
    if not reply_tokens:
        return []
    scored: list[tuple[float, str]] = []
    for result in retrieval.get("results", []):
        chunk = corpus_by_id.get(result["chunk_id"])
        if not chunk:
            continue
        chunk_tokens = set(tokenize(chunk["title"] + " " + chunk["text"], config))
        score = len(reply_tokens & chunk_tokens) / max(1, len(reply_tokens))
        if score >= threshold:
            scored.append((score, result["chunk_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk_id for _, chunk_id in scored[:max_ids]]


def _mode_connector(mode: str) -> str | None:
    if mode == "no_guarantees":
        return "I can't guarantee a specific outcome or timing."
    if mode == "manual_action_not_allowed":
        return "Support can't view, set, or reset your password."
    if mode == "review_required":
        return "This request requires review."
    return None


def _humanize_policy_sentence(sentence: str) -> str:
    text = sentence.strip()
    specific = [
        (
            "They may explain that the case will be reviewed and that users may be asked for additional information.",
            "Your case will be reviewed, and you may be asked for additional information.",
        ),
        (
            "They may explain that the case will be reviewed.",
            "Your case will be reviewed.",
        ),
        (
            "If the user should be asked to",
            "You may be asked to",
        ),
    ]
    for old, new in specific:
        text = text.replace(old, new)

    replacements = [
        (r"\bCustomers who forget a password can use\b", "If you forget your password, you can use"),
        (r"\bCustomers can\b", "You can"),
        (r"\bcustomers can\b", "you can"),
        (r"\bCustomers\b", "You"),
        (r"\bcustomers\b", "you"),
        (r"\bIf the user does not\b", "If you do not"),
        (r"\bif the user does not\b", "if you do not"),
        (r"\bthe user does not\b", "you do not"),
        (r"\bthe user is\b", "you are"),
        (r"\bthe user has\b", "you have"),
        (r"\bUsers can\b", "You can"),
        (r"\busers can\b", "you can"),
        (r"\bUsers may\b", "You may"),
        (r"\busers may\b", "you may"),
        (r"\bThe user should be asked to\b", "You may be asked to"),
        (r"\bthe user should be asked to\b", "you may be asked to"),
        (r"\bIf the user\b", "If you"),
        (r"\bif the user\b", "if you"),
        (r"\bthe user\b", "you"),
        (r"\bthe customer\b", "you"),
        (r"\btheir\b", "your"),
        (r"\bThey may\b", "Support may"),
        (r"\bthey should\b", "you should"),
        (r"\bSupport agents\b", "Support"),
        (r"\bsupport agents\b", "support"),
        (r"\bIf documents are rejected\b", "If your documents are rejected"),
        (r"\bask the customer to\b", "you should"),
        (r"\bask the user to\b", "you should"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    for pattern, replacement in [
        (r"\byou does\b", "you do"),
        (r"\byou has\b", "you have"),
        (r"\byou is\b", "you are"),
        (r"\byou was\b", "you were"),
        (r"\byou have not\b", "you have not"),
    ]:
        text = re.sub(pattern, replacement, text)

    text = normalize_whitespace(text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _is_internal_prohibition(sentence: str, mode: str) -> bool:
    if mode == "manual_action_not_allowed":
        return False
    lowered = sentence.lower()
    return (
        "must not" in lowered
        or "cannot promise" in lowered
        or "cannot guarantee" in lowered
        or lowered.startswith("support agents cannot")
        or "are not allowed" in lowered
    )


def _sentence_score(
    sentence: str,
    query_tokens: set[str],
    config: dict[str, Any],
    mode: str,
) -> float:
    sentence_tokens = set(tokenize(sentence, config))
    score = float(len(sentence_tokens & query_tokens))
    lowered = sentence.lower()
    if any(
        marker in lowered
        for marker in ("review", "depends", "typically", "may delay", "additional information", "eligib")
    ):
        score += 0.35
    if len(sentence) < 25:
        score -= 0.2
    return score


def select_policy_sentences(
    text: str,
    query_tokens: list[str],
    config: dict[str, Any],
    mode: str,
    max_sentences: int = 3,
) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    query_set = set(query_tokens)
    candidates = [
        (_sentence_score(sentence, query_set, config, mode), index, sentence)
        for index, sentence in enumerate(sentences)
        if not _is_internal_prohibition(sentence, mode)
    ]
    if not candidates:
        candidates = [
            (_sentence_score(sentence, query_set, config, mode), index, sentence)
            for index, sentence in enumerate(sentences)
        ]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:max_sentences]
    if not selected:
        selected = candidates[: min(2, len(candidates))]
    selected.sort(key=lambda item: item[1])
    return [item[2] for item in selected]


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def build_insufficient_fallback(
    message_id: str,
    gate: dict[str, Any],
    reason: str,
    generator: str = "safe_fallback",
    retrieved_ids: list[str] | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    mode = gate.get("policy_mode", "standard")
    opening = "I'm sorry for the inconvenience." if variant == "apology" else "Thanks for reaching out."
    if mode == "manual_action_not_allowed":
        reply = (
            opening
            + " Support can't view, set, or reset your password."
            + " I don't have an approved policy that covers the rest of this request, so I've escalated it "
            "to a support specialist for review."
        )
    elif mode == "no_guarantees":
        reply = (
            opening
            + " I don't have an approved policy that covers this request, so I've escalated it to a support "
            "specialist for review. I can't guarantee a specific outcome or timing."
        )
    elif mode == "review_required":
        reply = (
            opening
            + " I don't have an approved policy that covers this request. This request requires review, "
            "so I've escalated it to a support specialist."
        )
    else:
        reply = (
            opening
            + " I don't have an approved policy that covers this request, so I've escalated it to a support "
            "specialist for review. A support specialist will follow up with next steps."
        )
    return {
        "message_id": message_id,
        "reply": reply,
        "used_chunk_ids": [],
        "policy_mode": mode,
        "gate_decision_id": gate.get("gate_decision_id"),
        "gate_created_at": gate.get("created_at"),
        "retrieved_chunk_ids": unique_preserving_order(retrieved_ids or []),
        "generator": generator,
        "model": None,
        "llm_call_id": None,
        "citation_source": "none",
        "attempt": 1,
        "insufficient_policy_support": True,
        "fallback_reason": reason,
        "created_at": utc_now_iso(),
    }


def build_extractive_draft(
    message: dict[str, Any],
    retrieval: dict[str, Any],
    gate: dict[str, Any],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    generator: str = "extractive",
    reason: str | None = None,
    is_fallback: bool = False,
) -> dict[str, Any]:
    if retrieval.get("weak_support") or not retrieval.get("results"):
        return build_insufficient_fallback(
            message["message_id"],
            gate,
            reason or "insufficient_policy_support",
            generator=generator,
            retrieved_ids=retrieved_chunk_ids(retrieval),
        )

    top_result = retrieval["results"][0]
    chunk = corpus_by_id.get(top_result["chunk_id"], top_result)
    mode = gate.get("policy_mode", "standard")
    query_tokens = tokenize(message["customer_message"], config)
    selected = select_policy_sentences(
        chunk["text"],
        query_tokens=query_tokens,
        config=config,
        mode=mode,
        max_sentences=2 if mode == "standard" else 3,
    )
    humanized = [_humanize_policy_sentence(sentence) for sentence in selected]
    reply_parts = ["Thanks for reaching out."] + humanized
    reply = " ".join(part for part in reply_parts if part).strip()

    connector = _mode_connector(mode)
    if connector:
        mode_terms = {
            "no_guarantees": ["cannot", "can't", "unable", "not able", "not guaranteed", "no guarantee"],
            "manual_action_not_allowed": ["cannot", "can't", "unable", "not able"],
            "review_required": ["review", "additional information", "may ask", "check", "process"],
        }.get(mode, [])
        if not _contains_any(reply, mode_terms):
            reply = reply + " " + connector

    reply = normalize_whitespace(reply)
    used_chunk_ids = [top_result["chunk_id"]] if humanized or connector else []
    draft = {
        "message_id": message["message_id"],
        "reply": reply,
        "used_chunk_ids": used_chunk_ids,
        "policy_mode": mode,
        "gate_decision_id": gate.get("gate_decision_id"),
        "gate_created_at": gate.get("created_at"),
        "retrieved_chunk_ids": retrieved_chunk_ids(retrieval),
        "generator": generator,
        "model": None,
        "llm_call_id": None,
        "citation_source": "extractive",
        "attempt": 1,
        "insufficient_policy_support": False,
        "fallback_reason": reason,
        "created_at": utc_now_iso(),
    }
    if is_fallback:
        draft["generator"] = "safe_fallback"
    return draft


def build_safe_fallback(
    message: dict[str, Any],
    retrieval: dict[str, Any],
    gate: dict[str, Any],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    reason: str,
    variant: str | None = None,
) -> dict[str, Any]:
    if retrieval.get("weak_support") or not retrieval.get("results"):
        return build_insufficient_fallback(
            message["message_id"],
            gate,
            reason,
            retrieved_ids=retrieved_chunk_ids(retrieval),
            variant=variant,
        )
    return build_extractive_draft(
        message,
        retrieval,
        gate,
        corpus_by_id,
        config,
        generator="safe_fallback",
        reason=reason,
        is_fallback=True,
    )


def generate_draft(
    message: dict[str, Any],
    retrieval: dict[str, Any],
    gate: dict[str, Any],
    corpus_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
    llm_client: LLMClient | None,
    repair_context: dict[str, Any] | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    gate_created_at = gate.get("created_at")
    stage = "repair" if repair_context else "generate"

    if retrieval.get("weak_support") or not retrieval.get("results"):
        return build_insufficient_fallback(
            message["message_id"],
            gate,
            "insufficient_policy_support",
            retrieved_ids=retrieved_chunk_ids(retrieval),
        )

    if llm_client is not None and llm_client.available:
        system_prompt, user_prompt = build_prompts(
            message, retrieval, gate, config, repair_context=repair_context
        )
        result = llm_client.generate(
            system_prompt,
            user_prompt,
            stage=stage,
            input_artifacts=["policy_corpus.json", "retrieval_audit.json", "gate_decisions.json"],
            output_artifact="repair_actions.json" if repair_context else "draft_replies.json",
        )
        if result.ok:
            parsed = parse_json_object(result.text)
            if parsed is not None:
                allowed = retrieved_chunk_ids(retrieval)
                used = _filter_chunk_ids(parsed["used_chunk_ids"], allowed)
                citation_source = "model"
                if not used:
                    used = infer_used_chunk_ids(parsed["reply"], retrieval, corpus_by_id, config)
                    citation_source = "inferred"
                return {
                    "message_id": message["message_id"],
                    "reply": normalize_whitespace(parsed["reply"]),
                    "used_chunk_ids": used,
                    "policy_mode": gate.get("policy_mode", "standard"),
                    "gate_decision_id": gate.get("gate_decision_id"),
                    "gate_created_at": gate_created_at,
                    "retrieved_chunk_ids": allowed,
                    "generator": "llm",
                    "model": result.model,
                    "llm_call_id": result.call_id,
                    "citation_source": citation_source,
                    "attempt": attempt,
                    "insufficient_policy_support": False,
                    "fallback_reason": None,
                    "created_at": utc_now_iso(),
                }
            reason = "llm_json_parse_error"
        else:
            reason = "llm_error: " + str(result.error)

        if llm_client.required:
            raise RuntimeError(f"LLM generation failed and llm_required is true: {reason}")

        if repair_context:
            fallback = build_safe_fallback(
                message,
                retrieval,
                gate,
                corpus_by_id,
                config,
                reason="llm_repair_failed: " + reason,
            )
            fallback["attempt"] = attempt
            return fallback

        draft = build_extractive_draft(
            message,
            retrieval,
            gate,
            corpus_by_id,
            config,
            generator="extractive_after_llm_error",
            reason=reason,
        )
        draft["llm_error"] = reason
        draft["attempt"] = attempt
        return draft

    if repair_context:
        fallback = build_safe_fallback(
            message,
            retrieval,
            gate,
            corpus_by_id,
            config,
            reason="offline_targeted_repair",
        )
        fallback["attempt"] = attempt
        return fallback

    draft = build_extractive_draft(
        message,
        retrieval,
        gate,
        corpus_by_id,
        config,
        generator="extractive",
    )
    draft["attempt"] = attempt
    return draft


def corpus_index_by_id(corpus: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in corpus}


def draft_fingerprint(draft: dict[str, Any]) -> str:
    return short_hash(draft.get("reply", ""), 16)
