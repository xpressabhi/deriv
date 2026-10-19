"""End-to-end orchestration for the support-reply pipeline."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .config import api_key_source_kind, load_config
from .gate import decide_all, VALID_MODES
from .generate import (
    build_safe_fallback,
    corpus_index_by_id,
    generate_draft,
)
from .index import build_index
from .ingest import corpus_hash, ingest_documents
from .io_utils import (
    ensure_dir,
    normalize_whitespace,
    parse_iso,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    sha256_text,
    utc_now_iso,
    unique_preserving_order,
    write_json,
    write_text,
)
from .llm import LLMClient
from .report import build_operator_report
from .retrieve import retrieve_all, retrieved_chunk_ids
from .verify import verify_draft

ARTIFACTS = {
    "corpus": "policy_corpus.json",
    "retrieval": "retrieval_audit.json",
    "gates": "gate_decisions.json",
    "drafts": "draft_replies.json",
    "verifications": "verification_results.json",
    "repairs": "repair_actions.json",
    "finals": "final_responses.json",
    "llm_calls": "llm_calls.jsonl",
    "change_report": "chunk_change_report.json",
    "operator_md": "operator_report.md",
    "operator_json": "operator_report.json",
    "manifest": "run_manifest.json",
    "validation": "validation_report.json",
}


def _load_messages(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError("messages.json must be a non-empty JSON array")
    messages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Message at index {index} is not an object")
        message_id = str(item.get("message_id") or "").strip()
        customer_message = str(item.get("customer_message") or "").strip()
        if not message_id:
            raise ValueError(f"Message at index {index} has no message_id")
        if not customer_message:
            raise ValueError(f"Message {message_id} has no customer_message")
        if message_id in seen:
            raise ValueError(f"Duplicate message_id: {message_id}")
        seen.add(message_id)
        messages.append({"message_id": message_id, "customer_message": customer_message})
    return messages


def _final_from_draft(
    message: dict[str, Any],
    draft: dict[str, Any],
    retrieval: dict[str, Any],
    status: str,
    verification: dict[str, Any],
    repair_action: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    reply = normalize_whitespace(str(draft.get("reply", "")))
    return {
        "message_id": message["message_id"],
        "reply": reply,
        "status": status,
        "used_chunk_ids": unique_preserving_order(
            [str(chunk_id) for chunk_id in draft.get("used_chunk_ids", [])]
        ),
        "retrieved_chunk_ids": unique_preserving_order(
            draft.get("retrieved_chunk_ids") or retrieved_chunk_ids(retrieval)
        ),
        "policy_mode": draft.get("policy_mode"),
        "gate_decision_id": draft.get("gate_decision_id"),
        "verification_approved": bool(verification.get("approved")),
        "verification_attempt": verification.get("attempt"),
        "verification_draft_hash": verification.get("draft_hash"),
        "draft_generator": draft.get("generator"),
        "citation_source": draft.get("citation_source"),
        "repair_action": repair_action,
        "fallback_reason": fallback_reason,
        "insufficient_policy_support": bool(draft.get("insufficient_policy_support")),
        "final_hash": sha256_text(reply),
        "created_at": utc_now_iso(),
    }


def run_pipeline(
    policy_path: str | Path = "policy_docs.json",
    messages_path: str | Path = "messages.json",
    config_path: str | Path = "config.json",
    out_dir: str | Path = ".",
    offline: bool = False,
    no_cache: bool = False,
    llm_required: bool | None = None,
    draft_generator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy_file = Path(policy_path).resolve()
    messages_file = Path(messages_path).resolve()
    config_file = Path(config_path).resolve()
    output = ensure_dir(Path(out_dir).resolve())

    config = load_config(config_file)
    generation_config = config.setdefault("generation", {})
    if llm_required is not None:
        generation_config["llm_required"] = bool(llm_required)
    if offline:
        generation_config["llm_required"] = False

    if not policy_file.exists():
        raise FileNotFoundError(f"policy_docs.json not found: {policy_file}")
    if not messages_file.exists():
        raise FileNotFoundError(f"messages.json not found: {messages_file}")

    policy_payload = read_json(policy_file)
    messages = _load_messages(messages_file)
    corpus_by_id: dict[str, dict[str, Any]] = {}
    generator_fn = draft_generator or generate_draft

    run_id = "run-" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "") + "-" + uuid.uuid4().hex[:6]
    started_at = utc_now_iso()
    stage_records: list[dict[str, Any]] = []

    def begin_stage(name: str) -> dict[str, Any]:
        return {"stage": name, "started_at": utc_now_iso(), "_t0": time.monotonic()}

    def end_stage(record: dict[str, Any], details: dict[str, Any] | None = None, status: str = "ok") -> None:
        record["finished_at"] = utc_now_iso()
        record["duration_ms"] = int((time.monotonic() - record.pop("_t0")) * 1000)
        record["status"] = status
        record["details"] = details or {}
        stage_records.append(record)

    llm_client = LLMClient(
        config,
        out_dir=output,
        run_id=run_id,
        offline=offline,
        use_cache=not no_cache,
    )
    llm_client.check_ready()
    llm_client.start_run()

    # --- Ingest -----------------------------------------------------------
    stage = begin_stage("ingest")
    corpus = ingest_documents(policy_payload, config)
    corpus_by_id = corpus_index_by_id(corpus)
    write_json(output / ARTIFACTS["corpus"], corpus)
    end_stage(
        stage,
        {
            "documents": (
                len(policy_payload.get("documents", []))
                if isinstance(policy_payload, dict)
                else len(policy_payload)
            ),
            "chunks": len(corpus),
        },
    )

    # --- Index ------------------------------------------------------------
    stage = begin_stage("index")
    index_dir = output / str(config.get("artifacts", {}).get("index_dir", "index"))
    index, change_report = build_index(corpus, config, index_dir)
    write_json(output / ARTIFACTS["change_report"], change_report)
    end_stage(
        stage,
        {
            "chunks": len(corpus),
            "index_action": change_report.get("action"),
            "changed": change_report.get("changed"),
            "token_cache_hits": change_report.get("token_cache_hits"),
            "token_cache_misses": change_report.get("token_cache_misses"),
        },
    )

    # --- Retrieve ---------------------------------------------------------
    stage = begin_stage("retrieve")
    retrievals = retrieve_all(messages, index, config)
    write_json(output / ARTIFACTS["retrieval"], retrievals)
    end_stage(stage, {"messages": len(messages), "weak_support": sum(1 for r in retrievals if r["weak_support"])})

    # --- Deterministic gate ----------------------------------------------
    stage = begin_stage("gate")
    gates = decide_all(messages, config)
    write_json(output / ARTIFACTS["gates"], gates)
    gate_by_message = {gate["message_id"]: gate for gate in gates}
    end_stage(
        stage,
        {
            "messages": len(gates),
            "modes": dict(sorted(Counter(g["policy_mode"] for g in gates).items())),
            "sensitive": sum(1 for g in gates if g["sensitive"]),
        },
    )

    # --- Generate ---------------------------------------------------------
    stage = begin_stage("generate")
    retrieval_by_message = {retrieval["message_id"]: retrieval for retrieval in retrievals}
    drafts: list[dict[str, Any]] = []
    for message in messages:
        message_id = message["message_id"]
        gate = gate_by_message[message_id]
        if not gate.get("gate_decision_id"):
            raise RuntimeError(f"Generation attempted without a gate decision for {message_id}")
        if gate["policy_mode"] not in VALID_MODES:
            raise RuntimeError(f"Invalid policy mode for {message_id}: {gate['policy_mode']}")
        draft = generator_fn(
            message,
            retrieval_by_message[message_id],
            gate,
            corpus_by_id,
            config,
            llm_client,
            attempt=1,
        )
        if draft.get("gate_decision_id") != gate.get("gate_decision_id"):
            raise RuntimeError(f"Draft {message_id} does not reference its gate decision")
        if draft.get("policy_mode") != gate.get("policy_mode"):
            raise RuntimeError(f"Draft {message_id} policy_mode does not match the gate")
        if parse_iso(draft["created_at"]) < parse_iso(gate["created_at"]):
            raise RuntimeError(f"Draft {message_id} was created before its gate decision")
        drafts.append(draft)
    write_json(output / ARTIFACTS["drafts"], drafts)
    end_stage(stage, {"drafts": len(drafts), "generators": dict(sorted(Counter(d.get("generator") for d in drafts).items()))})

    # --- Verify -----------------------------------------------------------
    stage = begin_stage("verify")
    draft_by_message = {draft["message_id"]: draft for draft in drafts}
    verifications: list[dict[str, Any]] = []
    initial_verification: dict[str, dict[str, Any]] = {}
    for message in messages:
        message_id = message["message_id"]
        verification = verify_draft(
            draft_by_message[message_id],
            message,
            retrieval_by_message[message_id],
            gate_by_message[message_id],
            corpus_by_id,
            config,
            attempt=1,
            is_fallback=bool(draft_by_message[message_id].get("insufficient_policy_support")),
        )
        initial_verification[message_id] = verification
        verifications.append(verification)
    write_json(output / ARTIFACTS["verifications"], verifications)
    end_stage(
        stage,
        {
            "approved": sum(1 for v in verifications if v["approved"]),
            "failed": sum(1 for v in verifications if not v["approved"]),
            "issue_types": dict(
                sorted(
                    Counter(
                        issue["type"]
                        for verification in verifications
                        for issue in verification.get("issues", [])
                    ).items()
                )
            ),
        },
    )

    # --- Repair / fallback / finals --------------------------------------
    stage = begin_stage("repair")
    repair_records: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    max_repair_attempts = int(config.get("repair", {}).get("max_attempts", 1))

    for message in messages:
        message_id = message["message_id"]
        retrieval = retrieval_by_message[message_id]
        gate = gate_by_message[message_id]
        draft = draft_by_message[message_id]
        verification = initial_verification[message_id]

        if verification["approved"]:
            if draft.get("insufficient_policy_support"):
                reason = draft.get("fallback_reason") or "insufficient_policy_support"
                finals.append(
                    _final_from_draft(
                        message,
                        draft,
                        retrieval,
                        "safe_fallback",
                        verification,
                        repair_action="fallback",
                        fallback_reason=reason,
                    )
                )
                repair_records.append(
                    {
                        "message_id": message_id,
                        "action": "fallback",
                        "reason": reason,
                        "initial_draft_hash": sha256_text(draft.get("reply", "")),
                        "initial_issue_types": [],
                        "initial_issues": [],
                        "repair_attempted": False,
                        "fallback_reply_hash": sha256_text(draft.get("reply", "")),
                        "fallback_reply": draft.get("reply"),
                        "fallback_verification_approved": True,
                        "created_at": utc_now_iso(),
                    }
                )
                continue
            finals.append(
                _final_from_draft(
                    message, draft, retrieval, "approved", verification, repair_action="none", fallback_reason=None
                )
            )
            continue

        issue_types = sorted({issue["type"] for issue in verification.get("issues", [])})
        initial_hash = sha256_text(draft.get("reply", ""))

        if draft.get("insufficient_policy_support") or max_repair_attempts <= 0:
            reason = "insufficient_policy_support" if draft.get("insufficient_policy_support") else "repair_disabled"
            fallback = build_safe_fallback(
                message, retrieval, gate, corpus_by_id, config, reason=reason
            )
            if sha256_text(fallback.get("reply", "")) == initial_hash:
                fallback = build_safe_fallback(
                    message, retrieval, gate, corpus_by_id, config, reason=reason, variant="apology"
                )
            fallback["attempt"] = 2
            fallback_verification = verify_draft(
                fallback,
                message,
                retrieval,
                gate,
                corpus_by_id,
                config,
                attempt=2,
                is_fallback=True,
            )
            verifications.append(fallback_verification)
            finals.append(
                _final_from_draft(
                    message,
                    fallback,
                    retrieval,
                    "safe_fallback",
                    fallback_verification,
                    repair_action="fallback",
                    fallback_reason=reason,
                )
            )
            repair_records.append(
                {
                    "message_id": message_id,
                    "action": "fallback",
                    "reason": reason,
                    "initial_draft_hash": initial_hash,
                    "initial_issue_types": issue_types,
                    "initial_issues": verification.get("issues", []),
                    "repair_attempted": False,
                    "fallback_reply_hash": sha256_text(fallback.get("reply", "")),
                    "fallback_reply": fallback.get("reply"),
                    "fallback_verification_approved": fallback_verification["approved"],
                    "created_at": utc_now_iso(),
                }
            )
            continue

        repair_context = {"previous_reply": draft.get("reply", ""), "issues": verification.get("issues", [])}
        repaired = generator_fn(
            message,
            retrieval,
            gate,
            corpus_by_id,
            config,
            llm_client,
            repair_context=repair_context,
            attempt=2,
        )
        repaired["attempt"] = 2
        if repaired.get("gate_decision_id") != gate.get("gate_decision_id"):
            repaired["gate_decision_id"] = gate.get("gate_decision_id")
        repaired["policy_mode"] = gate.get("policy_mode")
        if parse_iso(repaired["created_at"]) < parse_iso(gate["created_at"]):
            raise RuntimeError(f"Repair {message_id} was created before its gate decision")
        repaired_verification = verify_draft(
            repaired,
            message,
            retrieval,
            gate,
            corpus_by_id,
            config,
            attempt=2,
        )
        verifications.append(repaired_verification)

        if repaired_verification["approved"] and sha256_text(repaired.get("reply", "")) != initial_hash:
            finals.append(
                _final_from_draft(
                    message,
                    repaired,
                    retrieval,
                    "repaired",
                    repaired_verification,
                    repair_action="regenerate",
                    fallback_reason=None,
                )
            )
            repair_records.append(
                {
                    "message_id": message_id,
                    "action": "regenerate",
                    "reason": "verification_failed_initial",
                    "initial_draft_hash": initial_hash,
                    "initial_issue_types": issue_types,
                    "initial_issues": verification.get("issues", []),
                    "repair_attempted": True,
                    "repaired_draft_hash": sha256_text(repaired.get("reply", "")),
                    "repaired_reply": repaired.get("reply"),
                    "repaired_verification_approved": True,
                    "created_at": utc_now_iso(),
                }
            )
            continue

        reason = "verification_failed_after_repair"
        fallback = build_safe_fallback(
            message, retrieval, gate, corpus_by_id, config, reason=reason
        )
        fallback["attempt"] = 3
        fallback_verification = verify_draft(
            fallback,
            message,
            retrieval,
            gate,
            corpus_by_id,
            config,
            attempt=3,
            is_fallback=True,
        )
        verifications.append(fallback_verification)
        finals.append(
            _final_from_draft(
                message,
                fallback,
                retrieval,
                "safe_fallback",
                fallback_verification,
                repair_action="fallback",
                fallback_reason=reason,
            )
        )
        repair_records.append(
            {
                "message_id": message_id,
                "action": "fallback",
                "reason": reason,
                "initial_draft_hash": initial_hash,
                "initial_issue_types": issue_types,
                "initial_issues": verification.get("issues", []),
                "repair_attempted": True,
                "repaired_draft_hash": sha256_text(repaired.get("reply", "")),
                "repaired_reply": repaired.get("reply"),
                "repaired_verification_approved": repaired_verification["approved"],
                "fallback_reply_hash": sha256_text(fallback.get("reply", "")),
                "fallback_reply": fallback.get("reply"),
                "fallback_verification_approved": fallback_verification["approved"],
                "created_at": utc_now_iso(),
            }
        )

    write_json(output / ARTIFACTS["verifications"], verifications)
    write_json(output / ARTIFACTS["repairs"], repair_records)
    write_json(output / ARTIFACTS["finals"], finals)
    end_stage(
        stage,
        {
            "finals": dict(sorted(Counter(final["status"] for final in finals).items())),
            "repair_actions": dict(sorted(Counter(record["action"] for record in repair_records).items())),
        },
    )

    # --- Operator report --------------------------------------------------
    stage = begin_stage("report")
    llm_calls = read_jsonl(output / ARTIFACTS["llm_calls"])
    operator_md, operator_json = build_operator_report(
        run_id=run_id,
        messages=messages,
        retrievals=retrievals,
        gates=gates,
        drafts=drafts,
        verifications=verifications,
        finals=finals,
        repair_records=repair_records,
        llm_calls=llm_calls,
        config=config,
    )
    write_text(output / ARTIFACTS["operator_md"], operator_md)
    write_json(output / ARTIFACTS["operator_json"], operator_json)
    end_stage(stage, {"issues": operator_json.get("verification_issue_type_counts", {})})

    # --- Manifest ---------------------------------------------------------
    finished_at = utc_now_iso()
    artifact_paths = {
        key: str(output / name)
        for key, name in ARTIFACTS.items()
        if key not in {"manifest", "validation"}
    }
    artifact_hashes: dict[str, str] = {}
    for key, path in artifact_paths.items():
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            artifact_hashes[key] = sha256_file(file_path)

    manifest = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "policy_path": str(policy_file),
        "messages_path": str(messages_file),
        "config_path": str(config_file),
        "config_hash": sha256_json(config),
        "input_hashes": {
            "policy_docs": sha256_file(policy_file),
            "messages": sha256_file(messages_file),
            "config": sha256_file(config_file),
        },
        "stage_order": [record["stage"] for record in stage_records],
        "stages": stage_records,
        "counts": {
            "messages": len(messages),
            "chunks": len(corpus),
            "final_statuses": dict(sorted(Counter(final["status"] for final in finals).items())),
            "policy_modes": dict(sorted(Counter(gate["policy_mode"] for gate in gates).items())),
        },
        "llm_mode": llm_client.mode(),
        "llm_model": llm_client.model if llm_client.available else None,
        "api_key_source": api_key_source_kind() if llm_client.available else None,
        "verifier": "deterministic",
        "gate_version": gates[0].get("gate_version") if gates else None,
        "corpus_hash": corpus_hash(corpus),
        "artifact_paths": artifact_paths,
        "artifact_hashes": artifact_hashes,
    }
    write_json(output / ARTIFACTS["manifest"], manifest)

    return {
        "run_id": run_id,
        "out_dir": str(output),
        "message_count": len(messages),
        "chunk_count": len(corpus),
        "final_statuses": manifest["counts"]["final_statuses"],
        "policy_modes": manifest["counts"]["policy_modes"],
        "llm_mode": manifest["llm_mode"],
        "artifacts": artifact_paths,
        "manifest_path": str(output / ARTIFACTS["manifest"]),
    }
