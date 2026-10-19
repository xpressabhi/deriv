"""Operator-facing summary of pipeline outcomes and recurring issues."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .io_utils import utc_now_iso


def _examples(verifications: list[dict[str, Any]], limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for verification in verifications:
        for issue in verification.get("issues", []):
            issue_type = issue.get("type", "unknown")
            bucket = grouped.setdefault(issue_type, [])
            if len(bucket) < limit:
                bucket.append(
                    {
                        "message_id": verification.get("message_id"),
                        "attempt": verification.get("attempt"),
                        "is_fallback": verification.get("is_fallback", False),
                        "text": issue.get("text"),
                        "supporting_chunk_ids": issue.get("supporting_chunk_ids", []),
                        "rule_id": issue.get("rule_id"),
                    }
                )
    return grouped


def build_operator_report(
    run_id: str,
    messages: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    drafts: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    finals: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    status_counts = Counter(final.get("status", "unknown") for final in finals)
    mode_counts = Counter(gate.get("policy_mode", "unknown") for gate in gates)
    weak_count = sum(1 for retrieval in retrievals if retrieval.get("weak_support"))
    fallback_reasons = Counter(
        final.get("fallback_reason")
        for final in finals
        if final.get("status") == "safe_fallback" and final.get("fallback_reason")
    )
    issue_counts = Counter()
    for verification in verifications:
        for issue in verification.get("issues", []):
            issue_counts[issue.get("type", "unknown")] += 1

    examples = _examples(verifications)
    llm_status = Counter(call.get("status", "unknown") for call in llm_calls)

    recommendations: list[str] = []
    if issue_counts.get("prohibited_promise"):
        recommendations.append(
            "Prohibited-promise findings appeared. Tighten the generation prompt, inspect the flagged "
            "examples, and keep the deterministic verifier as the release gate."
        )
    if issue_counts.get("unsupported_claim"):
        recommendations.append(
            "Unsupported claims were detected. Review retrieval quality and the grounding threshold; "
            "add policy coverage where the same gap repeats."
        )
    if issue_counts.get("policy_mismatch"):
        recommendations.append(
            "Some drafts diverged from the deterministic policy mode. Review mode-specific prompt wording "
            "and the verifier's required-language rules."
        )
    if weak_count:
        recommendations.append(
            f"{weak_count} message(s) had weak retrieval support. Consider adding or clarifying policy "
            "documents and extending the configured synonym map."
        )
    if fallback_reasons:
        recommendations.append(
            "Safe fallbacks were used. Sample them during review to confirm they are clear and do not "
            "over-promise, and track the recurring reasons."
        )
    if not recommendations:
        recommendations.append("No verification failures were observed in this run.")

    data = {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "message_count": len(messages),
        "final_status_counts": dict(sorted(status_counts.items())),
        "policy_mode_counts": dict(sorted(mode_counts.items())),
        "verification_issue_type_counts": dict(sorted(issue_counts.items())),
        "issue_examples": examples,
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        "weak_retrieval_count": weak_count,
        "llm_status_counts": dict(sorted(llm_status.items())),
        "draft_generator_counts": dict(
            sorted(Counter(draft.get("generator", "unknown") for draft in drafts).items())
        ),
        "repair_action_counts": dict(
            sorted(Counter(record.get("action", "unknown") for record in repair_records).items())
        ),
        "recommendations": recommendations,
    }

    lines: list[str] = [
        "# Operator Report",
        "",
        f"**Run:** `{run_id}`  ",
        f"**Generated:** {data['generated_at']}  ",
        f"**Messages:** {len(messages)}",
        "",
        "## Final outcomes",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status in ("approved", "repaired", "safe_fallback"):
        lines.append(f"| {status} | {status_counts.get(status, 0)} |")
    for status, count in sorted(status_counts.items()):
        if status not in {"approved", "repaired", "safe_fallback"}:
            lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## Deterministic gate modes",
            "",
            "| Mode | Count |",
            "| --- | ---: |",
        ]
    )
    for mode, count in sorted(mode_counts.items()):
        lines.append(f"| {mode} | {count} |")

    lines.extend(["", "## Verification issues by type", ""])
    if issue_counts:
        for issue_type, count in sorted(issue_counts.items()):
            lines.append(f"### {issue_type} ({count})")
            lines.append("")
            for example in examples.get(issue_type, []):
                chunk_text = ", ".join(example.get("supporting_chunk_ids") or []) or "none"
                lines.append(
                    f"- `{example.get('message_id')}` attempt {example.get('attempt')}: "
                    f"\"{example.get('text')}\" (support: {chunk_text})"
                )
            lines.append("")
    else:
        lines.extend(["No verification issues were recorded.", ""])

    lines.extend(["## Safe fallbacks", ""])
    if fallback_reasons:
        for reason, count in sorted(fallback_reasons.items()):
            lines.append(f"- **{reason}**: {count}")
    else:
        lines.append("No safe fallbacks were required.")
    lines.append("")

    lines.extend(
        [
            "## Retrieval and model usage",
            "",
            f"- Weak retrieval cases: {weak_count}",
            f"- LLM calls logged: {len(llm_calls)}",
            f"- LLM statuses: {dict(sorted(llm_status.items())) if llm_status else 'none (offline/extractive)'}",
            f"- Draft generators: {data['draft_generator_counts']}",
            f"- Repair actions: {data['repair_action_counts']}",
            "",
            "## Recommended operator actions",
            "",
        ]
    )
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")
    lines.append("")

    return "\n".join(lines), data
