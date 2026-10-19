"""Unit and end-to-end tests for the support-reply pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import validate as validate_module
from support_pipeline import config as config_module
from support_pipeline import llm as llm_module
from support_pipeline.gate import VALID_MODES, decide
from support_pipeline.generate import corpus_index_by_id, generate_draft
from support_pipeline.io_utils import (
    normalize_whitespace,
    read_json,
    read_jsonl,
    sha256_text,
    utc_now_iso,
    write_json,
)
from support_pipeline.llm import LLMClient, LLMResult
from support_pipeline.pipeline import run_pipeline
from support_pipeline.verify import verify_draft

BASE = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class PipelineTestBase(unittest.TestCase):
    def write_inputs(
        self,
        directory: Path,
        policy: object | None = None,
        messages: object | None = None,
        config: dict | None = None,
    ) -> tuple[Path, Path, Path]:
        policy_payload = policy if policy is not None else read_json(BASE / "policy_docs.json")
        message_payload = messages if messages is not None else read_json(BASE / "messages.json")
        config_payload = config if config is not None else read_json(BASE / "config.json")
        policy_path = directory / "policy_docs.json"
        messages_path = directory / "messages.json"
        config_path = directory / "config.json"
        write_json(policy_path, policy_payload)
        write_json(messages_path, message_payload)
        write_json(config_path, config_payload)
        return policy_path, messages_path, config_path

    def run_offline(
        self,
        directory: Path,
        policy: object | None = None,
        messages: object | None = None,
        config: dict | None = None,
        **kwargs: object,
    ) -> dict:
        policy_path, messages_path, config_path = self.write_inputs(directory, policy, messages, config)
        summary = run_pipeline(
            policy_path=policy_path,
            messages_path=messages_path,
            config_path=config_path,
            out_dir=directory,
            offline=True,
            **kwargs,
        )
        summary["_paths"] = (policy_path, messages_path, config_path)
        return summary

    def artifacts(self, directory: Path) -> dict:
        return {
            "corpus": read_json(directory / "policy_corpus.json"),
            "retrieval": read_json(directory / "retrieval_audit.json"),
            "gates": read_json(directory / "gate_decisions.json"),
            "drafts": read_json(directory / "draft_replies.json"),
            "verifications": read_json(directory / "verification_results.json"),
            "finals": read_json(directory / "final_responses.json"),
            "repairs": read_json(directory / "repair_actions.json"),
            "manifest": read_json(directory / "run_manifest.json"),
            "change_report": read_json(directory / "chunk_change_report.json"),
        }


class SampleEndToEndTests(PipelineTestBase):
    def test_sample_run_offline_is_valid_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            summary = self.run_offline(directory)
            self.assertEqual(summary["final_statuses"], {"approved": 4})
            self.assertEqual(summary["llm_mode"], "offline")

            policy_path, messages_path, config_path = summary["_paths"]
            report = validate_module.validate(directory, policy_path, messages_path, config_path)
            self.assertTrue(report["valid"], report["errors"])

            artifacts = self.artifacts(directory)
            gates = {gate["message_id"]: gate for gate in artifacts["gates"]}
            self.assertEqual(gates["M1"]["policy_mode"], "no_guarantees")
            self.assertEqual(gates["M2"]["policy_mode"], "no_guarantees")
            self.assertEqual(gates["M3"]["policy_mode"], "standard")
            self.assertEqual(gates["M4"]["policy_mode"], "no_guarantees")

            retrieval = {r["message_id"]: r for r in artifacts["retrieval"]}
            self.assertEqual(retrieval["M1"]["results"][0]["chunk_id"], "POL-002-c00")
            self.assertEqual(retrieval["M2"]["results"][0]["chunk_id"], "POL-003-c00")
            self.assertEqual(retrieval["M3"]["results"][0]["chunk_id"], "POL-001-c00")
            self.assertEqual(retrieval["M4"]["results"][0]["chunk_id"], "POL-004-c00")

            for final in artifacts["finals"]:
                self.assertIn(final["status"], {"approved", "repaired", "safe_fallback"})
                self.assertTrue(final["reply"].strip())
                self.assertTrue(final["used_chunk_ids"])
                self.assertNotIn("guaranteed approval", final["reply"].lower())

            self.assertEqual((directory / "llm_calls.jsonl").read_text(encoding="utf-8").strip(), "")
            self.assertTrue((directory / "operator_report.md").exists())
            operator = read_json(directory / "operator_report.json")
            self.assertEqual(operator["final_status_counts"], {"approved": 4})
            self.assertIn("## Verification issues by type", (directory / "operator_report.md").read_text(encoding="utf-8"))

    def test_corpus_chunk_fields_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.run_offline(directory)
            corpus = read_json(directory / "policy_corpus.json")
            self.assertTrue(corpus)
            required = {"chunk_id", "doc_id", "title", "chunk_index", "text", "content_hash"}
            ids = set()
            for chunk in corpus:
                self.assertTrue(required.issubset(chunk.keys()))
                self.assertNotIn(chunk["chunk_id"], ids)
                ids.add(chunk["chunk_id"])
                self.assertEqual(chunk["content_hash"], sha256_text(normalize_whitespace(chunk["text"])))

    def test_stage_order_proves_gate_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.run_offline(directory)
            manifest = read_json(directory / "run_manifest.json")
            order = manifest["stage_order"]
            self.assertLess(order.index("gate"), order.index("generate"))
            self.assertLess(order.index("generate"), order.index("verify"))
            self.assertLess(order.index("verify"), order.index("repair"))

    def test_offline_run_has_empty_llm_log_and_offline_manifest_mode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.run_offline(directory)
            manifest = read_json(directory / "run_manifest.json")
            self.assertEqual(manifest["llm_mode"], "offline")
            self.assertIsNone(manifest["llm_model"])
            self.assertEqual(read_jsonl(directory / "llm_calls.jsonl"), [])


class GateTests(PipelineTestBase):
    def setUp(self) -> None:
        self.config = read_json(BASE / "config.json")

    def _mode(self, message_id: str, text: str) -> str:
        return decide({"message_id": message_id, "customer_message": text}, self.config)["policy_mode"]

    def test_sensitive_guarantee_and_immediacy_language(self) -> None:
        self.assertEqual(
            self._mode("G1", "I submitted my ID yesterday. Can you confirm I will be verified today?"),
            "no_guarantees",
        )
        self.assertEqual(
            self._mode("G2", "Please refund my deposit right now. I need confirmation that the money is coming back."),
            "no_guarantees",
        )
        self.assertEqual(
            self._mode("G3", "Close my account today and make sure nothing blocks it."),
            "no_guarantees",
        )

    def test_standard_request_is_not_over_gated(self) -> None:
        self.assertEqual(
            self._mode("G4", "I forgot my password and I am not getting any reset email. What should I do?"),
            "standard",
        )

    def test_staff_password_action_is_manual_only(self) -> None:
        self.assertEqual(self._mode("G5", "Can you reset my password for me?"), "manual_action_not_allowed")
        self.assertEqual(self._mode("G6", "Tell me my password."), "manual_action_not_allowed")

    def test_curly_apostrophes_are_normalized_for_gate_matching(self) -> None:
        self.assertEqual(self._mode("G10", "Please reset the customer\u2019s password."), "manual_action_not_allowed")
        self.assertEqual(self._mode("G11", "Can\u2019t you refund me today?"), "no_guarantees")

    def test_sensitive_request_without_guarantee_requires_review(self) -> None:
        self.assertEqual(self._mode("G7", "What is your refund policy?"), "review_required")
        self.assertEqual(self._mode("G8", "How long does identity verification take?"), "review_required")

    def test_password_reset_link_request_is_not_a_staff_password_action(self) -> None:
        self.assertEqual(self._mode("G9", "Please send me the password reset email."), "standard")

    def test_all_modes_are_valid(self) -> None:
        for text in (
            "Please refund me today",
            "Can you reset my password?",
            "What is your refund policy?",
            "Thanks for the update",
        ):
            self.assertIn(self._mode("GX", text), VALID_MODES)


class VerifierTests(PipelineTestBase):
    def _context(self, directory: Path, message_id: str = "M2") -> tuple:
        summary = self.run_offline(directory)
        artifacts = self.artifacts(directory)
        message = next(
            m for m in read_json(directory / "messages.json") if m["message_id"] == message_id
        )
        retrieval = next(r for r in artifacts["retrieval"] if r["message_id"] == message_id)
        gate = next(g for g in artifacts["gates"] if g["message_id"] == message_id)
        corpus_by_id = corpus_index_by_id(artifacts["corpus"])
        config = read_json(directory / "config.json")
        return message, retrieval, gate, corpus_by_id, config

    def test_prohibited_promise_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory)
            draft = {
                "message_id": message["message_id"],
                "reply": "Your refund will be issued today and we guarantee it.",
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertFalse(result["approved"])
            self.assertIn("prohibited_promise", {issue["type"] for issue in result["issues"]})

    def test_grounded_hedged_reply_is_approved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory)
            draft = {
                "message_id": message["message_id"],
                "reply": (
                    "Refund eligibility depends on payment method, account status, and internal review. "
                    "Your case will be reviewed and you may be asked for additional information. "
                    "I can't guarantee a specific outcome."
                ),
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertTrue(result["approved"], result["issues"])

    def test_message_without_policy_support_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory)
            draft = {
                "message_id": message["message_id"],
                "reply": "We will send a courier to your home with a cheque tomorrow.",
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertFalse(result["approved"])
            self.assertIn("unsupported_claim", {issue["type"] for issue in result["issues"]})

    def test_invented_timeframe_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory, message_id="M1")
            draft = {
                "message_id": message["message_id"],
                "reply": "Identity verification will be completed within 5 business days.",
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertFalse(result["approved"])
            self.assertIn("unsupported_claim", {issue["type"] for issue in result["issues"]})

    def test_negated_closure_statement_is_not_a_promise(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory, message_id="M4")
            draft = {
                "message_id": message["message_id"],
                "reply": (
                    "Closure timing depends on open investigations, unresolved balances, or compliance "
                    "restrictions and may be delayed. Closure is not immediate unless confirmed by "
                    "back-office systems."
                ),
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertTrue(result["approved"], result["issues"])

    def test_courtesy_sentence_with_timing_is_not_a_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory, message_id="M4")
            draft = {
                "message_id": message["message_id"],
                "reply": (
                    "I understand you want to close your account today. Closure timing depends on open "
                    "investigations, unresolved balances, or compliance restrictions and may be delayed."
                ),
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertTrue(result["approved"], result["issues"])

    def test_curly_apostrophe_negation_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory, message_id="M4")
            draft = {
                "message_id": message["message_id"],
                "reply": (
                    "You can request account closure through support, but open investigations, unresolved "
                    "balances, or compliance restrictions may delay closure, and I can\u2019t confirm that "
                    "closure will be immediate or completed today."
                ),
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertTrue(result["approved"], result["issues"])

    def test_unable_to_guarantee_disclaimer_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            message, retrieval, gate, corpus_by_id, config = self._context(directory, message_id="M1")
            draft = {
                "message_id": message["message_id"],
                "reply": (
                    "Identity verification reviews are typically completed within 1 to 3 business days. "
                    "I'm not able to guarantee a specific result or completion date."
                ),
                "used_chunk_ids": [retrieval["results"][0]["chunk_id"]],
                "policy_mode": gate["policy_mode"],
            }
            result = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertTrue(result["approved"], result["issues"])


class FallbackTests(PipelineTestBase):
    def test_bad_generator_is_replaced_by_safe_fallback(self) -> None:
        def bad_generator(
            message: dict,
            retrieval: dict,
            gate: dict,
            corpus_by_id: dict,
            config: dict,
            llm_client: object,
            repair_context: dict | None = None,
            attempt: int = 1,
        ) -> dict:
            top_chunk = retrieval["results"][0]["chunk_id"]
            return {
                "message_id": message["message_id"],
                "reply": "Your refund will be issued today and we guarantee it.",
                "used_chunk_ids": [top_chunk],
                "policy_mode": gate["policy_mode"],
                "gate_decision_id": gate["gate_decision_id"],
                "gate_created_at": gate["created_at"],
                "retrieved_chunk_ids": [result["chunk_id"] for result in retrieval["results"]],
                "generator": "test_bad_generator",
                "model": None,
                "llm_call_id": None,
                "citation_source": "test",
                "attempt": attempt,
                "insufficient_policy_support": False,
                "fallback_reason": None,
                "created_at": utc_now_iso(),
            }

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            summary = self.run_offline(directory, draft_generator=bad_generator)
            self.assertEqual(summary["final_statuses"], {"safe_fallback": 4})
            artifacts = self.artifacts(directory)
            self.assertTrue(all(record["action"] == "fallback" for record in artifacts["repairs"]))
            for final in artifacts["finals"]:
                self.assertNotIn("guarantee it", final["reply"].lower())
                self.assertTrue(final["fallback_reason"])
            policy_path, messages_path, config_path = summary["_paths"]
            report = validate_module.validate(directory, policy_path, messages_path, config_path)
            self.assertTrue(report["valid"], report["errors"])


class LLMPathTests(PipelineTestBase):
    class FakeLLM:
        available = True
        required = False
        model = "fake-model"

        def __init__(self, text: str) -> None:
            self.text = text
            self.calls: list[str] = []

        def generate(self, system_prompt: str, user_prompt: str, stage: str, input_artifacts=None, output_artifact=None) -> LLMResult:
            self.calls.append(stage)
            return LLMResult(ok=True, text=self.text, call_id="fake-call-1", model=self.model)

    def test_llm_generation_path_uses_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.run_offline(directory)
            artifacts = self.artifacts(directory)
            message = next(m for m in read_json(directory / "messages.json") if m["message_id"] == "M2")
            retrieval = next(r for r in artifacts["retrieval"] if r["message_id"] == "M2")
            gate = next(g for g in artifacts["gates"] if g["message_id"] == "M2")
            corpus_by_id = corpus_index_by_id(artifacts["corpus"])
            config = read_json(directory / "config.json")
            top_chunk = retrieval["results"][0]["chunk_id"]
            fake = self.FakeLLM(
                json.dumps(
                    {
                        "reply": (
                            "Refund eligibility depends on payment method, account status, and internal review. "
                            "Your case will be reviewed and you may be asked for additional information. "
                            "I can't guarantee a specific outcome."
                        ),
                        "used_chunk_ids": [top_chunk],
                    }
                )
            )
            draft = generate_draft(message, retrieval, gate, corpus_by_id, config, fake, attempt=1)
            self.assertEqual(draft["generator"], "llm")
            self.assertEqual(draft["model"], "fake-model")
            self.assertEqual(draft["used_chunk_ids"], [top_chunk])
            verification = verify_draft(draft, message, retrieval, gate, corpus_by_id, config)
            self.assertTrue(verification["approved"], verification["issues"])

    def test_invalid_llm_json_falls_back_to_extractive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.run_offline(directory)
            artifacts = self.artifacts(directory)
            message = next(m for m in read_json(directory / "messages.json") if m["message_id"] == "M2")
            retrieval = next(r for r in artifacts["retrieval"] if r["message_id"] == "M2")
            gate = next(g for g in artifacts["gates"] if g["message_id"] == "M2")
            corpus_by_id = corpus_index_by_id(artifacts["corpus"])
            config = read_json(directory / "config.json")
            draft = generate_draft(
                message,
                retrieval,
                gate,
                corpus_by_id,
                config,
                self.FakeLLM("this is not json"),
                attempt=1,
            )
            self.assertEqual(draft["generator"], "extractive_after_llm_error")
            self.assertIn("llm_json_parse_error", draft.get("llm_error", ""))

    def test_llm_required_without_key_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            policy_path, messages_path, config_path = self.write_inputs(directory)
            with mock.patch.object(llm_module, "get_api_key", return_value=None), mock.patch.object(
                llm_module, "api_key_source", return_value=None
            ):
                with self.assertRaises(RuntimeError):
                    run_pipeline(
                        policy_path=policy_path,
                        messages_path=messages_path,
                        config_path=config_path,
                        out_dir=directory,
                        offline=False,
                        llm_required=True,
                    )

    def test_llm_client_parses_response_and_logs_without_secrets(self) -> None:
        class FakeResponse:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self) -> "LLMPathTests.FakeResponse":
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self) -> bytes:
                payload = {
                    "choices": [{"message": {"content": '{"reply": "ok", "used_chunk_ids": []}'}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                }
                return json.dumps(payload).encode("utf-8")

        captured: dict = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse()

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = read_json(BASE / "config.json")
            config["generation"]["use_cache"] = False
            with mock.patch.object(llm_module, "get_api_key", return_value="test-secret-key"), mock.patch.object(
                llm_module, "api_key_source", return_value="TEST_ENV"
            ):
                client = LLMClient(config, directory, "run-test", offline=False, use_cache=False)
                client.start_run()
                with mock.patch.object(llm_module.urllib.request, "urlopen", side_effect=fake_urlopen):
                    result = client.generate(
                        "system prompt",
                        "user prompt",
                        stage="generate",
                        input_artifacts=["policy_corpus.json"],
                        output_artifact="draft_replies.json",
                    )
            self.assertTrue(result.ok, result.error)
            self.assertEqual(json.loads(result.text)["reply"], "ok")
            calls = read_jsonl(directory / "llm_calls.jsonl")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["stage"], "generate")
            self.assertEqual(len(calls[0]["prompt_sha256"]), 64)
            self.assertNotIn("test-secret-key", json.dumps(calls))
            self.assertNotIn("Bearer", json.dumps(calls))
            self.assertTrue((captured.get("authorization") or "").startswith("Bearer "))


class ParaphrasedFixtureTests(PipelineTestBase):
    def test_paraphrased_fixtures_gate_and_retrieve_correctly(self) -> None:
        policy = read_json(FIXTURES / "policy_docs_paraphrased.json")
        messages = read_json(FIXTURES / "messages_paraphrased.json")
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            summary = self.run_offline(directory, policy=policy, messages=messages)
            artifacts = self.artifacts(directory)
            gates = {gate["message_id"]: gate for gate in artifacts["gates"]}
            self.assertEqual(gates["T1"]["policy_mode"], "no_guarantees")
            self.assertEqual(gates["T2"]["policy_mode"], "no_guarantees")
            self.assertEqual(gates["T3"]["policy_mode"], "standard")
            self.assertEqual(gates["T4"]["policy_mode"], "no_guarantees")

            retrieval = {r["message_id"]: r for r in artifacts["retrieval"]}
            self.assertEqual(retrieval["T1"]["results"][0]["chunk_id"], "P-103-c00")
            self.assertEqual(retrieval["T2"]["results"][0]["chunk_id"], "P-102-c00")
            self.assertEqual(retrieval["T3"]["results"][0]["chunk_id"], "P-101-c00")
            self.assertEqual(retrieval["T4"]["results"][0]["chunk_id"], "P-104-c00")

            for final in artifacts["finals"]:
                self.assertTrue(final["reply"].strip())
                self.assertTrue(final["used_chunk_ids"])
            policy_path, messages_path, config_path = summary["_paths"]
            report = validate_module.validate(directory, policy_path, messages_path, config_path)
            self.assertTrue(report["valid"], report["errors"])

    def test_paraphrased_run_without_matching_gate_still_handles_generic_message(self) -> None:
        policy = read_json(FIXTURES / "policy_docs_paraphrased.json")
        messages = [{"message_id": "T5", "customer_message": "What are your weekend opening hours?"}]
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            summary = self.run_offline(directory, policy=policy, messages=messages)
            artifacts = self.artifacts(directory)
            self.assertEqual(artifacts["gates"][0]["policy_mode"], "standard")
            self.assertTrue(artifacts["retrieval"][0]["weak_support"])
            final = artifacts["finals"][0]
            self.assertEqual(final["status"], "safe_fallback")
            self.assertIn("insufficient_policy_support", final["fallback_reason"])
            self.assertTrue(final["retrieved_chunk_ids"])
            policy_path, messages_path, config_path = summary["_paths"]
            report = validate_module.validate(directory, policy_path, messages_path, config_path)
            self.assertTrue(report["valid"], report["errors"])


class IndexReuseTests(PipelineTestBase):
    def test_second_run_reuses_index_when_corpus_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.run_offline(directory)
            self.run_offline(directory)
            report = read_json(directory / "chunk_change_report.json")
            self.assertFalse(report["changed"])
            self.assertEqual(report["action"], "reused")

    def test_policy_change_rebuilds_only_affected_token_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.run_offline(directory)
            policy = read_json(directory / "policy_docs.json")
            for document in policy["documents"]:
                if document["doc_id"] == "POL-003":
                    document["text"] += " Internal payment operations may take additional time."
            write_json(directory / "policy_docs.json", policy)
            self.run_offline(directory, policy=policy)
            report = read_json(directory / "chunk_change_report.json")
            self.assertTrue(report["changed"])
            self.assertEqual(report["action"], "rebuilt_corpus_changed")
            self.assertIn("POL-003-c00", report["modified_chunk_ids"])
            self.assertGreaterEqual(report["token_cache_hits"], 1)
            self.assertGreaterEqual(report["token_cache_misses"], 1)


class ValidationFailureTests(PipelineTestBase):
    def test_validation_fails_when_required_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            summary = self.run_offline(directory)
            policy_path, messages_path, config_path = summary["_paths"]
            (directory / "final_responses.json").unlink()
            report = validate_module.validate(directory, policy_path, messages_path, config_path)
            self.assertFalse(report["valid"])
            self.assertTrue(any("final_responses" in error for error in report["errors"]))

    def test_validation_fails_when_inputs_change_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            summary = self.run_offline(directory)
            policy_path, messages_path, config_path = summary["_paths"]
            messages = read_json(messages_path)
            messages.append({"message_id": "M99", "customer_message": "A new question"})
            write_json(messages_path, messages)
            report = validate_module.validate(directory, policy_path, messages_path, config_path)
            self.assertFalse(report["valid"])


class DotenvTests(unittest.TestCase):
    def test_dotenv_key_is_loaded_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dotenv = Path(raw) / ".env"
            dotenv.write_text("OPENCODE_API_KEY=from-dotenv\n", encoding="utf-8")
            with mock.patch.object(config_module, "DOTENV_PATH", dotenv), mock.patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(config_module.get_api_key(), "from-dotenv")
                self.assertEqual(config_module.api_key_source(), "OPENCODE_API_KEY (.env)")
                self.assertEqual(config_module.api_key_source_kind(), "dotenv")

    def test_environment_variable_wins_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dotenv = Path(raw) / ".env"
            dotenv.write_text("OPENCODE_API_KEY=from-dotenv\n", encoding="utf-8")
            with mock.patch.object(config_module, "DOTENV_PATH", dotenv), mock.patch.dict(
                os.environ, {"OPENCODE_API_KEY": "from-env"}, clear=True
            ):
                self.assertEqual(config_module.get_api_key(), "from-env")


if __name__ == "__main__":
    unittest.main()
