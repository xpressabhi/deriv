# Grounded Support-Reply Pipeline

A small, replayable pipeline that turns short customer messages into draft
support responses using only approved policy snippets, with deterministic safety
checks that run before any reply is returned.

The pipeline is designed for evaluation from a clean checkout:

- runs with the Python standard library only
- works with no API key and no network (deterministic extractive generator)
- uses an LLM when an OpenCode Go / OpenAI-compatible key is available
- keeps generation and verification as separate stages
- writes structured, auditable artifacts on every run

The full implementation contract lives in [`SPEC.md`](SPEC.md).

---

## Quick start

```bash
python run.py          # regenerate every artifact from policy_docs.json + messages.json
python validate.py     # check artifacts, traceability, and safety invariants
python -m unittest discover -s tests -v
```

or:

```bash
make run
make validate
make test
```

The offline path is always available and needs no secrets:

```bash
python run.py --offline
```

---

## Using the LLM path

The generator calls an OpenAI-compatible chat endpoint. The default target is
OpenCode Go with DeepSeek V4.1 Flash:

```bash
export OPENCODE_API_KEY=your-key        # OPENCODE_GO_API_KEY also works
python run.py
```

Or put the key in a local, gitignored `.env` file (start from `.env.example`):

```bash
cp .env.example .env
# then edit .env and set OPENCODE_API_KEY
```

Behavior:

- With a key, the pipeline calls the model at temperature 0, parses strict JSON,
  cites only retrieved chunk IDs, and logs every call to `llm_calls.jsonl`.
- The client sends a normal `User-Agent` and the `x-opencode-session` header the
  Go endpoint requires for routing; both are configurable under `generation`.
- Without a key, it automatically uses the deterministic extractive generator.
- `--offline` forces the deterministic path even when a key is present.
- `--llm-required` (or `SUPPORT_LLM_REQUIRED=1`) fails fast instead of silently
  falling back.

Configuration is in `config.json`; environment overrides:

| Variable | Purpose |
| --- | --- |
| `OPENCODE_API_KEY` / `OPENCODE_GO_API_KEY` / `OPENCODE_ZEN_API_KEY` | Provider key |
| `OPENCODE_GO_BASE_URL` | Endpoint override |
| `SUPPORT_MODEL` | Model override |
| `SUPPORT_LLM_REQUIRED` | Fail if no key (`1`/`true`) |

No key is ever written to an artifact, cache file, or log.

---

## Pipeline stages

```text
policy_docs.json ──► ingest ──► policy_corpus.json
                        │
                        ▼
                      BM25 index ──► index/ (index + token cache + manifest)
                        │
messages.json ──────────┼──► retrieve ──► retrieval_audit.json
                        │
                        ▼
                 deterministic gate ──► gate_decisions.json
                        │
                        ▼
                   generate ──► draft_replies.json
                        │
                        ▼
                deterministic verify ──► verification_results.json
                        │
              approved ─┴─ failed
                 │            │
                 │            ▼
                 │      targeted repair or safe fallback ──► repair_actions.json
                 │            │
                 └──────► final_responses.json
                              │
                              ▼
              validate.py · report.py · run_manifest.json
```

### 1. Ingest and chunk

Policy documents are split on sentence boundaries into small chunks with
one-sentence overlap. Every chunk has `chunk_id`, `doc_id`, `title`,
`chunk_index`, `text`, and a SHA-256 `content_hash`.

### 2. Retrieval

A stdlib BM25 index retrieves the top 3 chunks per message. The index is stored
on disk under `index/` together with a token cache keyed by `content_hash`, so an
unchanged corpus reuses cached tokenization and the index itself.

### 3. Deterministic gate

Rule-based code (never a model) inspects the message for sensitive areas and
escalation language:

| Condition | Policy mode |
| --- | --- |
| Refund / verification / closure + guarantee or immediacy language | `no_guarantees` |
| Request for staff to view, set, reset, or reveal a password | `manual_action_not_allowed` |
| Sensitive area without guarantee language | `review_required` |
| Everything else | `standard` |

The decision, matched rules, and constraints are persisted to
`gate_decisions.json` before the generation stage starts. The runner refuses to
generate a draft without a matching gate decision.

### 4. Generation

The model receives the customer message, the authoritative gate result, and the
retrieved excerpts only. It must return:

```json
{"reply": "...", "used_chunk_ids": ["..."]}
```

Chunk IDs are validated against the retrieval result; invented IDs are dropped.
If the model returns no usable citations, the pipeline infers supporting chunks
from the reply text and records `citation_source: "inferred"`. If the call
fails or returns invalid JSON, the deterministic extractive generator takes over
and the reason is recorded on the draft.

### 5. Verification (separate stage)

The verifier is deterministic and never calls a model. It checks:

- citation integrity (non-empty, valid, retrieved IDs only)
- prohibited promises and staff password actions (negation-aware)
- grounding of every sentence against the cited chunks
- invented numbers and timeframes
- policy-mode compliance and required hedging language

It emits the required `approved` / `issues` structure with
`unsupported_claim`, `prohibited_promise`, or `policy_mismatch` issue types.

### 6. Repair or safe fallback

A failed draft triggers one targeted regeneration that includes the verifier
issues. The repaired draft is verified again in a separate verification record.
If it still fails, or if policy support is insufficient, the pipeline returns a
safe fallback that avoids commitments and states the request needs review. The
chosen path, reason, and hashes are recorded in `repair_actions.json`. A failed
draft is never returned as the final answer.

### 7. Validation and reporting

`validate.py` checks required artifacts, JSON shapes, coverage, traceability,
stage ordering (gate before generation), fallback documentation, and that no
failed draft was returned. It writes `validation_report.json` and exits nonzero
on failure.

`operator_report.md` / `operator_report.json` group failures by issue type and
list recommendations. `chunk_change_report.json` records added, modified,
removed, and unchanged chunks between runs.

---

## Generated artifacts

| Artifact | Contents |
| --- | --- |
| `policy_corpus.json` | Chunked corpus with content hashes |
| `index/` | BM25 index, token cache, index manifest |
| `retrieval_audit.json` | Top-3 results, scores, matched terms, chunk metadata |
| `gate_decisions.json` | Deterministic policy mode, matched rules, constraints |
| `draft_replies.json` | Initial draft per message with cited chunk IDs |
| `verification_results.json` | Per-attempt verifier output and issue lists |
| `repair_actions.json` | Regenerate vs fallback path and reason |
| `final_responses.json` | Final safe reply, status, traceability, hashes |
| `llm_calls.jsonl` | Stage, timestamp, prompt hash, artifacts, status, usage |
| `run_manifest.json` | Stage order, input/config hashes, artifact hashes |
| `validation_report.json` | Machine-readable validation result |
| `operator_report.md` / `.json` | Operator summary grouped by issue type |
| `chunk_change_report.json` | Content-hash based change detection |

Sample run with the offline generator: M1–M4 are all approved with
`no_guarantees` modes for M1/M2/M4 and `standard` for M3.

---

## Configuration

`config.json` controls retrieval (`top_k`, BM25, synonyms), chunking, gate
categories and patterns, escalation terms, verifier patterns and thresholds,
mode-required language, repair limits, and model settings. Guardrails are
configuration-driven so equivalent fixtures can be supported without code
changes; see `SPEC.md` for the full contract.

---

## Repository layout

```text
run.py                     # single-command runner
validate.py                # artifact and safety validation
config.json                # all guardrail and model settings
policy_docs.json           # sample input
messages.json              # sample input
support_pipeline/
  config.py                # config loading and env overrides
  io_utils.py              # atomic writes, hashing, JSON/JSONL helpers
  ingest.py                # chunking and corpus hashing
  index.py                 # BM25 index and token cache
  retrieve.py              # top-k retrieval and audit records
  gate.py                  # deterministic risk gate
  llm.py                   # OpenCode Go client, retries, cache, logging
  generate.py              # prompt construction, LLM + extractive generation
  verify.py                # deterministic compliance verifier
  pipeline.py              # orchestration and artifact writing
  report.py                # operator report
tests/
  test_pipeline.py         # unit and end-to-end tests
  fixtures/                # paraphrased policy/message fixtures
```

---

## Safety invariants

- The LLM never decides whether a prohibited guarantee is allowed.
- Gating and verification are deterministic code.
- Verification is a separate stage from generation.
- Every final reply is traceable to retrieved chunks, or is a documented
  insufficient-support fallback.
- Failed drafts are never returned as finals.
- The pipeline runs end-to-end with no secrets and no network.

---

## Extending with an LLM verifier

The verifier is intentionally a pure function over the draft, retrieved chunks,
and gate decision. To add an optional LLM semantic check later, call it from the
same `verify_draft` stage, merge its findings into the issue list, and keep the
deterministic checks as the veto layer.
