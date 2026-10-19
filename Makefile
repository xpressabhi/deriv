PYTHON ?= python3

.PHONY: run validate test all clean

run:
	$(PYTHON) run.py

validate:
	$(PYTHON) validate.py

test:
	$(PYTHON) -m unittest discover -s tests -v

all: run validate

clean:
	rm -rf index cache policy_corpus.json retrieval_audit.json gate_decisions.json draft_replies.json verification_results.json repair_actions.json final_responses.json llm_calls.jsonl run_manifest.json operator_report.md operator_report.json validation_report.json chunk_change_report.json
	rm -rf support_pipeline/__pycache__ tests/__pycache__
