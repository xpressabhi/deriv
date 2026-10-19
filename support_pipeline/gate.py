"""Deterministic policy risk gate.

This module deliberately contains no model calls. The gate runs in code before
generation and produces the policy mode that constrains the draft.
"""

from __future__ import annotations

import re
from typing import Any

from .io_utils import short_hash, utc_now_iso

GATE_VERSION = "1.0"
VALID_MODES = {"standard", "no_guarantees", "manual_action_not_allowed", "review_required"}


def _matches(text: str, patterns: list[str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for index, pattern in enumerate(patterns, start=1):
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid gate pattern {pattern!r}: {exc}") from exc
        match = compiled.search(text)
        if match:
            matches.append(
                {
                    "pattern_index": index,
                    "pattern": pattern,
                    "matched_text": match.group(0),
                    "span_start": match.start(),
                    "span_end": match.end(),
                }
            )
    return matches


def decide(
    message: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    message_id = message["message_id"]
    text = message["customer_message"]
    match_text = text.replace("\u2019", "'").replace("\u2018", "'")
    gate_config = config.get("gate", {})

    category_matches: dict[str, list[dict[str, str]]] = {}
    for category, settings in gate_config.get("categories", {}).items():
        found = _matches(match_text, settings.get("patterns", []))
        if found:
            category_matches[category] = found

    guarantee_matches = _matches(match_text, gate_config.get("guarantee_patterns", []))
    immediacy_matches = _matches(match_text, gate_config.get("immediacy_patterns", []))

    sensitive = bool(category_matches)
    escalation = bool(guarantee_matches or immediacy_matches)
    categories = sorted(category_matches)

    if "password_staff_action" in category_matches:
        policy_mode = "manual_action_not_allowed"
        constraints = ["manual_action_not_allowed"]
        if escalation:
            constraints.append("no_guarantees")
    elif sensitive and escalation:
        policy_mode = "no_guarantees"
        constraints = ["no_guarantees", "review_required"]
    elif sensitive:
        policy_mode = "review_required"
        constraints = ["review_required"]
    else:
        policy_mode = "standard"
        constraints = []

    matched_rules: list[dict[str, Any]] = []
    for category in categories:
        for match in category_matches[category]:
            matched_rules.append(
                {
                    "rule_id": f"category.{category}.p{match['pattern_index']}",
                    "kind": "category",
                    "category": category,
                    **match,
                }
            )
    for match in guarantee_matches:
        matched_rules.append(
            {
                "rule_id": f"escalation.guarantee.p{match['pattern_index']}",
                "kind": "escalation",
                "category": "guarantee",
                **match,
            }
        )
    for match in immediacy_matches:
        matched_rules.append(
            {
                "rule_id": f"escalation.immediacy.p{match['pattern_index']}",
                "kind": "escalation",
                "category": "immediacy",
                **match,
            }
        )

    reason_bits: list[str] = []
    if categories:
        reason_bits.append("sensitive areas: " + ", ".join(categories))
    if guarantee_matches:
        reason_bits.append("guarantee/confirmation language")
    if immediacy_matches:
        reason_bits.append("immediacy language")
    if not reason_bits:
        reason_bits.append("no sensitive category or escalation language matched")

    return {
        "gate_decision_id": "GATE-" + message_id + "-" + short_hash(text, 10),
        "gate_version": GATE_VERSION,
        "message_id": message_id,
        "customer_message": text,
        "input_hash": short_hash(text, 16),
        "sensitive": sensitive,
        "categories": categories,
        "escalation": escalation,
        "guarantee_hits": guarantee_matches,
        "immediacy_hits": immediacy_matches,
        "policy_mode": policy_mode,
        "constraints": constraints,
        "matched_rule_ids": [rule["rule_id"] for rule in matched_rules],
        "matched_rules": matched_rules,
        "reason": "; ".join(reason_bits),
        "created_at": utc_now_iso(),
    }


def decide_all(messages: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [decide(message, config) for message in messages]
