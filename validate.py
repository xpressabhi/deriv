"""Artifact and safety validation: python validate.py"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from support_pipeline.gate import VALID_MODES
from support_pipeline.io_utils import (
    normalize_whitespace,
    parse_iso,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)

BASE_DIR = Path(__file__).resolve().parent
VALID_STATUSES = {"approved", "repaired", "safe_fallback"}
VALID_ISSUE_TYPES = {"unsupported_claim", "prohibited_promise", "policy_mismatch"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_ARTIFACTS = {
    "policy_corpus": "policy_corpus.json",
    "retrieval_audit": "retrieval_audit.json",
    "gate_decisions": "gate_decisions.json",
    "draft_replies": "draft_replies.json",
    "verification_results": "verification_results.json",
    "repair_actions": "repair_actions.json",
    "final_responses": "final_responses.json",
    "llm_calls": "llm_calls.jsonl",
    "run_manifest": "run_manifest.json",
    "operator_report": "operator_report.md",
    "operator_report_json": "operator_report.json",
    "chunk_change_report": "chunk_change_report.json",
    "index": "index/bm25_index.json",
    "index_manifest": "index/index_manifest.json",
    "token_cache": "index/chunk_token_cache.json",
}

JSON_ARTIFACTS = {
    "policy_corpus": "policy_corpus.json",
    "retrieval_audit": "retrieval_audit.json",
    "gate_decisions": "gate_decisions.json",
    "draft_replies": "draft_replies.json",
    "verification_results": "verification_results.json",
    "repair_actions": "repair_actions.json",
    "final_responses": "final_responses.json",
    "run_manifest": "run_manifest.json",
    "operator_report_json": "operator_report.json",
    "chunk_change_report": "chunk_change_report.json",
    "index": "index/bm25_index.json",
    "index_manifest": "index/index_manifest.json",
    "token_cache": "index/chunk_token_cache.json",
}


def _index_by(records: list[dict[str, Any]], key: str = "message_id") -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if isinstance(record, dict) and record.get(key):
            indexed.setdefault(str(record[key]), []).append(record)
    return indexed


def _one_by(records: list[dict[str, Any]], key: str = "message_id") -> dict[str, dict[str, Any]]:
    return {message_id: items[0] for message_id, items in _index_by(records, key).items()}


def _parse_timestamp(value: Any) -> Any:
    try:
        return parse_iso(str(value))
    except (ValueError, TypeError):
        return None


def validate(
    out_dir: str | Path = ".",
    policy_path: str | Path = BASE_DIR / "policy_docs.json",
    messages_path: str | Path = BASE_DIR / "messages.json",
    config_path: str | Path = BASE_DIR / "config.json",
) -> dict[str, Any]:
    output = Path(out_dir).resolve()
    policy_file = Path(policy_path).resolve()
    messages_file = Path(messages_path).resolve()
    config_file = Path(config_path).resolve()

    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def check(name: str, passed: bool, details: dict[str, Any] | None = None) -> bool:
        check_record = {"name": name, "passed": bool(passed), "details": details or {}}
        checks.append(check_record)
        if not passed:
            errors.append(name + (": " + json.dumps(details, default=str) if details else ""))
        return bool(passed)

    # 1. Required files.
    for name, relative in REQUIRED_ARTIFACTS.items():
        path = output / relative
        check(f"artifact.exists.{name}", path.exists(), {"path": str(path)})
    if errors:
        return _report(output, checks, errors, {})

    # 2. Parse JSON artifacts.
    data: dict[str, Any] = {}
    for name, relative in JSON_ARTIFACTS.items():
        path = output / relative
        try:
            data[name] = read_json(path)
            check(f"artifact.json.{name}", True, {"path": str(path)})
        except (ValueError, OSError) as exc:
            check(f"artifact.json.{name}", False, {"path": str(path), "error": str(exc)})
    if errors:
        return _report(output, checks, errors, {})

    corpus = data["policy_corpus"]
    retrievals = data["retrieval_audit"]
    gates = data["gate_decisions"]
    drafts = data["draft_replies"]
    verifications = data["verification_results"]
    repairs = data["repair_actions"]
    finals = data["final_responses"]
    manifest = data["run_manifest"]

    for name, value, expected in (
        ("policy_corpus", corpus, list),
        ("retrieval_audit", retrievals, list),
        ("gate_decisions", gates, list),
        ("draft_replies", drafts, list),
        ("verification_results", verifications, list),
        ("repair_actions", repairs, list),
        ("final_responses", finals, list),
        ("run_manifest", manifest, dict),
    ):
        check(f"artifact.shape.{name}", isinstance(value, expected), {"actual": type(value).__name__})
    if errors:
        return _report(output, checks, errors, {})

    try:
        messages = read_json(messages_file)
    except (ValueError, OSError) as exc:
        check("input.messages", False, {"error": str(exc), "path": str(messages_file)})
        return _report(output, checks, errors, {})
    check("input.messages", isinstance(messages, list) and bool(messages))
    if not isinstance(messages, list) or not messages:
        return _report(output, checks, errors, {})
    message_ids = [str(message.get("message_id")) for message in messages if isinstance(message, dict)]
    message_set = set(message_ids)
    check("messages.ids.unique", len(message_ids) == len(message_set), {"count": len(message_ids)})
    check(
        "messages.fields",
        all(
            isinstance(message, dict)
            and str(message.get("message_id", "")).strip()
            and str(message.get("customer_message", "")).strip()
            for message in messages
        ),
    )

    # 3. Corpus.
    chunk_ids = [str(chunk.get("chunk_id")) for chunk in corpus if isinstance(chunk, dict)]
    chunk_set = set(chunk_ids)
    chunk_fields = {"chunk_id", "doc_id", "title", "chunk_index", "text", "content_hash"}
    check("corpus.nonempty", bool(corpus))
    check("corpus.ids.unique", len(chunk_ids) == len(chunk_set), {"count": len(chunk_ids)})
    check(
        "corpus.fields",
        all(isinstance(chunk, dict) and chunk_fields.issubset(chunk.keys()) for chunk in corpus),
    )
    check(
        "corpus.content_hashes",
        all(
            isinstance(chunk, dict)
            and isinstance(chunk.get("content_hash"), str)
            and bool(HEX64.match(chunk["content_hash"]))
            for chunk in corpus
        ),
    )
    check(
        "corpus.content_hash_matches_text",
        all(
            sha256_text(normalize_whitespace(str(chunk.get("text", "")))) == chunk.get("content_hash")
            for chunk in corpus
            if isinstance(chunk, dict)
        ),
    )
    check(
        "corpus.chunk_indices",
        all(isinstance(chunk.get("chunk_index"), int) and chunk["chunk_index"] >= 0 for chunk in corpus if isinstance(chunk, dict)),
    )

    # 4. Retrieval.
    retrieval_by_message = _one_by(retrievals)
    check("retrieval.coverage", message_set.issubset(set(retrieval_by_message)), {"messages": len(message_ids)})
    retrieval_ok = True
    retrieval_details: list[str] = []
    for message_id, retrieval in retrieval_by_message.items():
        results = retrieval.get("results", [])
        if not isinstance(results, list) or len(results) > 3:
            retrieval_ok = False
            retrieval_details.append(f"{message_id}: more than 3 results")
            continue
        scores = [float(result.get("score", 0)) for result in results if isinstance(result, dict)]
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            retrieval_ok = False
            retrieval_details.append(f"{message_id}: scores not descending")
        for result in results:
            if not isinstance(result, dict) or result.get("chunk_id") not in chunk_set:
                retrieval_ok = False
                retrieval_details.append(f"{message_id}: result chunk not in corpus")
        if retrieval.get("returned_count") != len(results):
            retrieval_ok = False
            retrieval_details.append(f"{message_id}: returned_count mismatch")
    check("retrieval.top3_and_valid_chunks", retrieval_ok, {"problems": retrieval_details})
    check(
        "retrieval.metadata",
        all(
            isinstance(result, dict)
            and {"chunk_id", "doc_id", "title", "score", "content_hash", "text"}.issubset(result.keys())
            for retrieval in retrievals
            for result in retrieval.get("results", [])
            if isinstance(retrieval, dict)
        ),
    )

    # 5. Gate.
    gate_by_message = _one_by(gates)
    check("gate.coverage", message_set.issubset(set(gate_by_message)))
    check("gate.one_per_message", all(len(items) == 1 for items in _index_by(gates).values()))
    check(
        "gate.valid_modes",
        all(gate.get("policy_mode") in VALID_MODES for gate in gates if isinstance(gate, dict)),
        {"modes": sorted({gate.get("policy_mode") for gate in gates if isinstance(gate, dict)})},
    )
    check(
        "gate.has_matched_rules",
        all(isinstance(gate.get("matched_rules"), list) for gate in gates if isinstance(gate, dict)),
    )

    stage_order = manifest.get("stage_order", []) if isinstance(manifest, dict) else []
    for required_stage in ("ingest", "index", "retrieve", "gate", "generate", "verify", "repair", "report"):
        check(f"manifest.stage.{required_stage}", required_stage in stage_order, {"stage_order": stage_order})
    if "gate" in stage_order and "generate" in stage_order:
        check("manifest.gate_before_generate", stage_order.index("gate") < stage_order.index("generate"))
    if "generate" in stage_order and "verify" in stage_order:
        check("manifest.generate_before_verify", stage_order.index("generate") < stage_order.index("verify"))

    # 6. Drafts.
    draft_by_message = _one_by(drafts)
    check("drafts.coverage", message_set.issubset(set(draft_by_message)))
    check("drafts.one_per_message", all(len(items) == 1 for items in _index_by(drafts).values()))
    drafts_ok = True
    draft_problems: list[str] = []
    for message_id, draft in draft_by_message.items():
        gate = gate_by_message.get(message_id, {})
        retrieval = retrieval_by_message.get(message_id, {})
        retrieved_ids = {result.get("chunk_id") for result in retrieval.get("results", [])}
        if not str(draft.get("reply", "")).strip():
            drafts_ok = False
            draft_problems.append(f"{message_id}: empty reply")
        if draft.get("gate_decision_id") != gate.get("gate_decision_id"):
            drafts_ok = False
            draft_problems.append(f"{message_id}: gate id mismatch")
        if draft.get("policy_mode") != gate.get("policy_mode"):
            drafts_ok = False
            draft_problems.append(f"{message_id}: policy mode mismatch")
        used = set(draft.get("used_chunk_ids", []))
        if not used.issubset(retrieved_ids):
            drafts_ok = False
            draft_problems.append(f"{message_id}: used chunks not retrieved")
        gate_time = _parse_timestamp(gate.get("created_at"))
        draft_time = _parse_timestamp(draft.get("created_at"))
        if gate_time is None or draft_time is None or draft_time < gate_time:
            drafts_ok = False
            draft_problems.append(f"{message_id}: draft not created after gate")
    check("drafts.grounded_and_gated", drafts_ok, {"problems": draft_problems})

    # 7. Verifications.
    initial_by_message: dict[str, dict[str, Any]] = {}
    for verification in verifications:
        message_id = verification.get("message_id")
        if verification.get("attempt") == 1 and message_id not in initial_by_message:
            initial_by_message[str(message_id)] = verification
    check("verifications.initial_coverage", message_set.issubset(set(initial_by_message)))
    verification_ok = True
    verification_problems: list[str] = []
    for verification in verifications:
        issues = verification.get("issues", [])
        if not isinstance(verification.get("approved"), bool):
            verification_ok = False
            verification_problems.append(f"{verification.get('message_id')}: approved is not bool")
        if not isinstance(issues, list):
            verification_ok = False
            verification_problems.append(f"{verification.get('message_id')}: issues is not a list")
            continue
        if verification.get("approved") and issues:
            verification_ok = False
            verification_problems.append(f"{verification.get('message_id')}: approved with issues")
        if not verification.get("approved") and not issues:
            verification_ok = False
            verification_problems.append(f"{verification.get('message_id')}: rejected without issues")
        for issue in issues:
            if issue.get("type") not in VALID_ISSUE_TYPES:
                verification_ok = False
                verification_problems.append(f"{verification.get('message_id')}: invalid issue type")
            if not isinstance(issue.get("supporting_chunk_ids"), list):
                verification_ok = False
                verification_problems.append(f"{verification.get('message_id')}: supporting ids not a list")
    check("verifications.structured", verification_ok, {"problems": verification_problems})
    check(
        "verifications.initial_hash_matches_draft",
        all(
            initial_by_message[message_id].get("draft_hash")
            == sha256_text(str(draft_by_message[message_id].get("reply", "")))
            for message_id in message_set & set(initial_by_message) & set(draft_by_message)
        ),
    )

    # 8. Finals.
    final_by_message = _one_by(finals)
    check("finals.coverage", message_set == set(final_by_message), {"messages": len(message_ids), "finals": len(finals)})
    check("finals.one_per_message", len(finals) == len(message_set), {"finals": len(finals)})
    final_ok = True
    final_problems: list[str] = []
    failed_hashes_by_message: dict[str, set[str]] = {}
    for verification in verifications:
        if not verification.get("approved"):
            failed_hashes_by_message.setdefault(str(verification.get("message_id")), set()).add(
                str(verification.get("draft_hash"))
            )

    for message_id in sorted(message_set & set(final_by_message)):
        final = final_by_message[message_id]
        gate = gate_by_message.get(message_id, {})
        retrieval = retrieval_by_message.get(message_id, {})
        retrieved_ids = {result.get("chunk_id") for result in retrieval.get("results", [])}
        used = set(final.get("used_chunk_ids", []))
        reply = str(final.get("reply", ""))
        status = final.get("status")

        if status not in VALID_STATUSES:
            final_ok = False
            final_problems.append(f"{message_id}: invalid status {status!r}")
        if not reply.strip():
            final_ok = False
            final_problems.append(f"{message_id}: empty final reply")
        if final.get("gate_decision_id") != gate.get("gate_decision_id"):
            final_ok = False
            final_problems.append(f"{message_id}: final gate id mismatch")
        if final.get("policy_mode") != gate.get("policy_mode"):
            final_ok = False
            final_problems.append(f"{message_id}: final policy mode mismatch")
        if not used.issubset(retrieved_ids):
            final_ok = False
            final_problems.append(f"{message_id}: final used chunks not retrieved")
        if final.get("final_hash") != sha256_text(reply):
            final_ok = False
            final_problems.append(f"{message_id}: final hash mismatch")
        if status in {"approved", "repaired"}:
            if not final.get("verification_approved"):
                final_ok = False
                final_problems.append(f"{message_id}: {status} final is not verification-approved")
            if not used:
                final_ok = False
                final_problems.append(f"{message_id}: {status} final has no used chunk IDs")
            matching = [
                verification
                for verification in verifications
                if verification.get("message_id") == message_id
                and verification.get("draft_hash") == final.get("verification_draft_hash")
                and verification.get("approved")
            ]
            if not matching:
                final_ok = False
                final_problems.append(f"{message_id}: no approved verification for final draft")
        if status == "safe_fallback":
            if not final.get("fallback_reason"):
                final_ok = False
                final_problems.append(f"{message_id}: safe fallback lacks a reason")
            if not used and not (final.get("insufficient_policy_support") and final.get("retrieved_chunk_ids")):
                final_ok = False
                final_problems.append(f"{message_id}: safe fallback lacks traceable retrieved chunks")
        if final.get("final_hash") in failed_hashes_by_message.get(message_id, set()):
            final_ok = False
            final_problems.append(f"{message_id}: final reply equals a failed draft")
        initial = initial_by_message.get(message_id)
        if initial and not initial.get("approved") and status != "approved":
            initial_draft = draft_by_message.get(message_id, {})
            if final.get("final_hash") == sha256_text(str(initial_draft.get("reply", ""))):
                final_ok = False
                final_problems.append(f"{message_id}: failed initial draft was returned")
    check("finals.safe_and_traceable", final_ok, {"problems": final_problems})

    # 9. Repair records line up with failure paths.
    repair_actions = {str(record.get("message_id")): record for record in repairs if isinstance(record, dict)}
    repair_ok = True
    repair_problems: list[str] = []
    for message_id, final in final_by_message.items():
        status = final.get("status")
        record = repair_actions.get(message_id)
        if status == "repaired" and (not record or record.get("action") != "regenerate"):
            repair_ok = False
            repair_problems.append(f"{message_id}: repaired final lacks regenerate record")
        if status == "safe_fallback" and (not record or record.get("action") != "fallback"):
            repair_ok = False
            repair_problems.append(f"{message_id}: fallback final lacks fallback record")
    check("repairs.paths_recorded", repair_ok, {"problems": repair_problems})

    # 10. LLM log.
    try:
        llm_calls = read_jsonl(output / "llm_calls.jsonl")
        llm_log_ok = True
        llm_problems: list[str] = []
        for call in llm_calls:
            for field in ("stage", "timestamp", "prompt_sha256", "input_artifacts", "output_artifact"):
                if field not in call:
                    llm_log_ok = False
                    llm_problems.append(f"missing {field}")
            if call.get("prompt_sha256") and not HEX64.match(str(call["prompt_sha256"])):
                llm_log_ok = False
                llm_problems.append("invalid prompt hash")
            serialized = json.dumps(call).lower()
            for secret_marker in ("authorization", "bearer ", "opencode_api_key", "sk-"):
                if secret_marker in serialized:
                    llm_log_ok = False
                    llm_problems.append(f"possible secret marker {secret_marker!r}")
        check("llm_calls.logged", llm_log_ok, {"calls": len(llm_calls), "problems": llm_problems})
        if manifest.get("llm_mode") == "llm":
            check("llm_calls.present_when_llm_mode", len(llm_calls) > 0, {"calls": len(llm_calls)})
    except (ValueError, OSError) as exc:
        check("llm_calls.readable", False, {"error": str(exc)})

    # 11. Manifest input hashes and stage timestamps.
    input_hashes = manifest.get("input_hashes", {}) if isinstance(manifest, dict) else {}
    check(
        "manifest.policy_hash_matches_current_input",
        input_hashes.get("policy_docs") == sha256_file(policy_file),
        {"path": str(policy_file)},
    )
    check(
        "manifest.messages_hash_matches_current_input",
        input_hashes.get("messages") == sha256_file(messages_file),
        {"path": str(messages_file)},
    )
    check(
        "manifest.config_hash_matches_current_input",
        input_hashes.get("config") == sha256_file(config_file),
        {"path": str(config_file)},
    )
    stage_records = manifest.get("stages", []) if isinstance(manifest, dict) else []
    for stage_name in ("gate", "generate", "verify", "repair"):
        matching = [record for record in stage_records if record.get("stage") == stage_name]
        check(f"manifest.stage_record.{stage_name}", len(matching) == 1)
    if len(stage_order) >= 2:
        gate_time = next(
            (_parse_timestamp(record.get("finished_at")) for record in stage_records if record.get("stage") == "gate"),
            None,
        )
        generate_time = next(
            (_parse_timestamp(record.get("started_at")) for record in stage_records if record.get("stage") == "generate"),
            None,
        )
        check(
            "manifest.gate_stage_finishes_before_generation",
            gate_time is not None and generate_time is not None and gate_time <= generate_time,
            {"gate_finished_at": str(gate_time), "generate_started_at": str(generate_time)},
        )

    summary = {
        "messages": len(message_set),
        "chunks": len(corpus),
        "retrievals": len(retrievals),
        "gates": len(gates),
        "drafts": len(drafts),
        "verifications": len(verifications),
        "repairs": len(repairs),
        "finals": len(finals),
        "final_statuses": dict(sorted(Counter(final.get("status") for final in finals).items())),
        "policy_modes": dict(sorted(Counter(gate.get("policy_mode") for gate in gates).items())),
        "issue_types": dict(
            sorted(
                Counter(
                    issue.get("type")
                    for verification in verifications
                    for issue in verification.get("issues", [])
                ).items()
            )
        ),
        "llm_calls": len(llm_calls) if "llm_calls" in locals() else 0,
    }
    return _report(output, checks, errors, summary)


def _report(
    output: Path,
    checks: list[dict[str, Any]],
    errors: list[str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "valid": not errors,
        "checked_at": utc_now_iso(),
        "out_dir": str(output),
        "checks": checks,
        "errors": errors,
        "summary": summary,
    }
    write_json(output / "validation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate support-reply pipeline artifacts")
    parser.add_argument("--out-dir", default=str(BASE_DIR), help="Directory containing generated artifacts")
    parser.add_argument("--policy", default=str(BASE_DIR / "policy_docs.json"))
    parser.add_argument("--messages", default=str(BASE_DIR / "messages.json"))
    parser.add_argument("--config", default=str(BASE_DIR / "config.json"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(
        out_dir=args.out_dir,
        policy_path=args.policy,
        messages_path=args.messages,
        config_path=args.config,
    )
    if not args.quiet:
        print(f"Validation: {'PASS' if report['valid'] else 'FAIL'}")
        print(f"  Checks: {len(report['checks'])}")
        print(f"  Errors: {len(report['errors'])}")
        if report.get("summary"):
            print(f"  Summary: {report['summary']}")
        if report["errors"]:
            print("")
            for error in report["errors"][:30]:
                print(f"  - {error}")
            if len(report["errors"]) > 30:
                print(f"  ... {len(report['errors']) - 30} more")
        print(f"  Report: {Path(args.out_dir).resolve() / 'validation_report.json'}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
