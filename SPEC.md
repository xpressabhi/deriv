# Support Reply Pipeline — Implementation Specification

This file is the source of truth for the implementation. It is intentionally kept
in the repository so completeness can be checked against the same criteria the
evaluator is likely to apply.

Status legend: `[ ]` not implemented, `[~]` implemented but not yet verified,
`[x]` implemented and verified.

---

## 1. Goal

Build a small, replayable support-reply pipeline that turns short customer
messages into draft support responses using only approved policy snippets.

The pipeline must enforce deterministic safety checks before any LLM-generated
reply is returned, keep generation and verification as separate stages, and make
every decision auditable through structured artifacts.

The evaluator may delete generated files and rerun the pipeline, and may replace
sample policy content and inbound messages with equivalent fixtures. No response
may be hardcoded to the sample inputs.

---

## 2. Inputs

### 2.1 `policy_docs.json`

```json
{
  "documents": [
    {"doc_id": "POL-001", "title": "Password Reset Policy", "text": "..."}
  ]
}
```

### 2.2 `messages.json`

```json
[
  {"message_id": "M1", "customer_message": "..."}
]
```

Both files must be read from disk on every run. Do not embed the sample content
in application code.

---

## 3. Required outputs

The pipeline must regenerate these artifacts from the input files:

| Artifact | Purpose |
| --- | --- |
| `policy_corpus.json` | Chunked policy corpus |
| `index/` | Local file-based retrieval index and token cache |
| `retrieval_audit.json` | Top-k retrieval results with scores and metadata |
| `gate_decisions.json` | Deterministic safety-gate decisions, one per message |
| `draft_replies.json` | Initial grounded draft, one per message |
| `verification_results.json` | Separate verifier output for every draft attempt |
| `repair_actions.json` | Repair vs fallback path and reason |
| `final_responses.json` | Safe final response, one per message |
| `llm_calls.jsonl` | Model call audit log |
| `run_manifest.json` | Stage order, input/config hashes, artifact index |
| `validation_report.json` | Machine-readable completeness/validation result |
| `operator_report.md` | Operator-facing failure summary |
| `operator_report.json` | Structured operator summary |
| `chunk_change_report.json` | Added/modified/removed chunk detection |

`llm_calls.jsonl` may be empty when the deterministic offline generator is used.

---

## 4. Functional requirements

### R1 — Policy ingestion and chunking `[x]`

- Load every document from `policy_docs.json`.
- Split policy text into small logical chunks suitable for retrieval.
- Each chunk must contain:
  - `chunk_id`
  - `doc_id`
  - `title`
  - `chunk_index`
  - `text`
  - `content_hash`
- Save the corpus to `policy_corpus.json`.
- Chunk IDs must be unique and stable for unchanged content.

### R2 — Retrieval over policy chunks `[x]`

- Retrieve the top 3 most relevant chunks for each customer message.
- Use a local, file-based retrieval index or embedding cache.
- Store similarity scores and retrieved chunk metadata.
- Save retrieval results to `retrieval_audit.json`.
- Do not hardcode responses or retrieval results to sample inputs.
- Handle weak retrieval explicitly.

### R3 — Deterministic policy risk gate `[x]`

- Run before any generation call.
- Use rule-based code only; no model judgment.
- Detect requests for guaranteed outcomes or immediate action in sensitive areas:
  - refund approval
  - verification approval
  - account-closure timing
  - password handling staff are not allowed to perform
- Emit a structured policy mode:
  - `no_guarantees`
  - `manual_action_not_allowed`
  - `review_required`
  - `standard`
- Persist the decision, matched rules, and constraints.
- Generation must be unable to proceed without a matching gate decision.

### R4 — Grounded reply generation `[x]`

- Generate one initial draft per message using only retrieved policy chunks and
  the gate policy mode.
- Avoid unsupported promises and actions.
- Return a structured object containing:
  - `message_id`
  - `reply`
  - `used_chunk_ids`
  - `policy_mode`
- Cite only chunk IDs that were retrieved.
- Support an LLM path through an OpenAI-compatible provider.
- Support a deterministic extractive path so a clean checkout without secrets
  still produces valid artifacts.

### R5 — Separate compliance verification `[x]`

- Run as a distinct stage after generation.
- Deterministic verifier only (no LLM for now).
- Detect:
  - unsupported guarantees
  - prohibited actions
  - claims not supported by retrieved chunks
  - policy-mode violations
  - invalid or missing citations
  - invented numbers/timeframes
- Output per verification attempt:
  - `message_id`
  - `approved`
  - `issues` with `type`, `text`, `supporting_chunk_ids`
- Allowed issue types: `unsupported_claim`, `prohibited_promise`,
  `policy_mismatch`.
- If verification fails, the original draft must not be returned.

### R6 — Targeted repair or safe fallback `[x]`

- If verification fails, either:
  - regenerate with a stricter prompt listing the verifier issues, then verify
    again; or
  - return a safe fallback that requires review and avoids commitments.
- Record the chosen path and reason in `repair_actions.json`.
- Never return a failed draft as the final response.

### R7 — Replayable runner `[x]`

- A single command (`python run.py` or `make run`) processes all messages and
  regenerates artifacts from input files.
- The command works from a clean checkout.
- Generated artifacts are written atomically.

### R8 — Validation command `[x]`

- Validate that required artifacts exist.
- Validate JSON/JSONL syntax and required fields.
- Validate that every message has an approved final reply or a documented safe
  fallback.
- Validate that final replies reference retrieved chunk IDs (or are documented
  insufficient-support fallbacks).
- Validate that deterministic gating ran before generation.
- Validate that failed drafts are not returned as finals.
- Write `validation_report.json` and exit nonzero on failure.

### R9 — Prompt and model call logging `[x]`

- Log each model call with:
  - stage
  - timestamp
  - prompt hash
  - input artifact references
  - output artifact reference
  - status, latency, usage, retries
- Never log API keys or secrets.
- Cache successful responses by prompt hash when LLM mode is active.

### R10 — Configurable thresholds and modes `[x]`

- Guardrail keyword lists, modes, retrieval settings, verification thresholds,
  and repair limits are configurable via `config.json`.
- Environment overrides are allowed for provider/model/API key.
- No secrets are embedded in code or config.

### R11 — Operator explanation summary `[x]`

- Produce a small operator-facing report.
- Group failures by issue type.
- Include repeated patterns and recommended actions.
- Write `operator_report.md` and `operator_report.json`.

### R12 — Chunk change detection `[x]`

- Detect changed chunks using `content_hash`.
- Report added, removed, modified, and unchanged chunks.
- Reuse cached tokenization for unchanged chunks.
- Rebuild the retrieval index when the corpus or retrieval config changes.
- Write `chunk_change_report.json`.

---

## 5. Safety and non-functional constraints

- No secrets in source code.
- No external website scraping.
- No managed vector database.
- The LLM must never decide whether a prohibited guarantee is allowed.
- Final replies must be traceable to retrieved policy chunks.
- Verification must be separate from generation.
- Deterministic gate must run before generation.
- Pipeline must run without a network or API key and still produce safe output.
- LLM use must be optional, not required, for a clean-checkout run.

---

## 6. Architecture

```text
policy_docs.json ──► ingest.py ──► policy_corpus.json
                                      │
                                      ▼
                                  index.py ──► index/ (BM25 + token cache)
                                      │
messages.json ────────────────────────┼──► retrieve.py ──► retrieval_audit.json
                                      │
                                      ▼
                                   gate.py ──► gate_decisions.json
                                      │
                                      ▼
                                 generate.py ──► draft_replies.json
                                      │
                                      ▼
                                  verify.py ──► verification_results.json
                                      │
                         approved ────┴──── failed
                            │                 │
                            │                 ▼
                            │            repair.py / safe fallback
                            │                 │
                            └────────► final_responses.json
                                      │
                                      ▼
                         validate.py + report.py + run_manifest.json
```

The deterministic gate and verifier never call a model. The LLM client is an
optional adapter behind `generate.py`; the extractive generator is the fallback.

---

## 7. Configuration contract

`config.json` must cover:

- retrieval top-k, BM25 parameters, stopwords, synonyms, minimum score
- chunk size and overlap
- sensitive categories and keyword/regex lists
- guarantee and immediacy escalation patterns
- prohibited verifier patterns and issue types
- minimum sentence grounding support
- mode-specific required language
- repair attempt limit
- model endpoint, model name, user agent, session header, temperature,
  timeout, retries
- cache and artifact directories

Environment variables:

- `OPENCODE_API_KEY` or `OPENCODE_GO_API_KEY` — provider key (environment or a local
  gitignored `.env` file)
- `OPENCODE_GO_BASE_URL` — endpoint override
- `SUPPORT_MODEL` — model override
- `SUPPORT_LLM_REQUIRED` — set to `1` to fail without an LLM
- `SUPPORT_OFFLINE` is not required; use `python run.py --offline`

---

## 8. Acceptance checklist

### Inputs and core pipeline

- [x] Sample `policy_docs.json` exists and matches the fixture shape
- [x] Sample `messages.json` exists and matches the fixture shape
- [x] Chunking produces required chunk fields
- [x] Retrieval returns top 3 with scores and metadata
- [x] Local file-based index exists
- [x] Deterministic gate emits policy mode before generation
- [x] Generation uses retrieved chunks and cites used chunk IDs
- [x] Verification is a separate deterministic stage
- [x] Failed drafts are repaired or replaced by safe fallback
- [x] Single-command runner works from a clean checkout
- [x] Validation command checks artifact and safety invariants
- [x] Model-call log records prompt hashes and artifacts
- [x] Guardrail settings are configurable
- [x] Operator report groups failures by issue type
- [x] Chunk change detection reports added/modified/removed chunks

### Robustness

- [x] Paraphrased policy/message fixtures work without code changes
- [x] LLM path tested against the live provider with a real key
- [x] Offline path tested with no API key and no network
- [x] `--offline` forces the deterministic generator
- [x] `SUPPORT_LLM_REQUIRED=1` fails clearly when no key is present
- [x] API keys never appear in artifacts or logs
- [x] Weak retrieval produces a documented safe fallback
- [x] Invalid model JSON falls back safely
- [x] Number/timeframe claims are grounded or rejected
- [x] Verifier rejects an injected prohibited-promise draft
- [x] Final response hash differs from a failed draft hash
- [x] `python -m unittest discover -s tests -v` passes

### Evaluation commands

```bash
python run.py
python validate.py
python -m unittest discover -s tests -v
```

or:

```bash
make run
make validate
make test
```

---

## 9. Implementation notes / decisions

- Retrieval is BM25 implemented in the standard library. This keeps the clean
  checkout dependency-free and deterministic.
- The verifier is deterministic. The code structure leaves room for an optional
  LLM verifier later, but the deterministic verifier remains authoritative.
- The gate and verifier use configuration-driven regexes so equivalent fixtures
  can be supported without code changes.
- The extractive generator quotes and lightly rewrites approved policy sentences.
  It is a safety fallback, not a replacement for the LLM path.
- LLM responses are cached by prompt hash only when the LLM is available, so an
  offline run never serves stale model output.
- The `run_manifest.json` stage timestamps are the proof that the gate stage
  completed before the generation stage started.
- `final_responses.json` records `status` (`approved`, `repaired`, or
  `safe_fallback`), verification reference, and the final reply hash.

---

## 10. Definition of done

The implementation is complete when:

1. All required artifacts are regenerated by `python run.py`.
2. `python validate.py` exits 0 and reports no errors.
3. The unit test suite passes.
4. The sample run produces safe final responses for M1–M4.
5. A paraphrased fixture run also produces safe final responses.
6. The checklist in section 8 is fully checked.

---

## 11. Verification evidence (last run)

Environment: macOS, Python 3 standard library only.

Offline path (no secrets, no network):

```text
$ python3 run.py --offline
Pipeline complete
  Messages:       4
  Chunks:         4
  Final statuses: {'approved': 4}
  Policy modes:   {'no_guarantees': 3, 'standard': 1}
  LLM mode:       offline

$ python3 -m unittest discover -s tests -v
Ran 32 tests
OK
```

Live provider path (OpenCode Go, `deepseek-v4.1-flash`, key in a gitignored
`.env`):

```text
$ python3 run.py
Pipeline complete
  Messages:       4
  Chunks:         4
  Final statuses: {'approved': 4}
  Policy modes:   {'no_guarantees': 3, 'standard': 1}
  LLM mode:       llm

$ python3 validate.py
Validation: PASS
  Checks: 82
  Errors: 0
  Summary: {'messages': 4, 'chunks': 4, 'finals': 4,
            'final_statuses': {'approved': 4},
            'policy_modes': {'no_guarantees': 3, 'standard': 1},
            'issue_types': {}, 'llm_calls': 4}
```

Live run notes:

- All four replies were generated by the model (`generator: llm`), passed the
  separate deterministic verifier, and cited their retrieved chunk IDs.
- An earlier live run produced an M4 draft that the verifier rejected with one
  `prohibited_promise` and one `unsupported_claim`; the targeted repair pass
  regenerated it and a second, separate verification approved the result. That
  run demonstrated the repair path end to end with the real model.
- Two verifier false positives found during the live run were fixed: negation
  inside the matched span (for example "closure is **not** immediate"), and
  courtesy acknowledgement sentences containing timing words. A second fresh
  live run exposed curly-apostrophe negation ("can’t") and the disclaimer
  phrasing "not able to guarantee"; both are now handled and covered by tests.
- `llm_calls.jsonl` recorded five successful HTTP 200 calls in the repairing
  run (four drafts plus one repair), with prompt hashes, usage, reasoning-token
  counts, and no secrets. Subsequent runs hit the prompt-hash cache and all four
  LLM drafts pass the deterministic verifier.
