# Common tasks. `make help` lists them. The docker targets are UNVERIFIED until Docker has been installed and used.
PY ?= python
OFFLINE = LLM_PROVIDER=mock SEARCH_PROVIDER=offline

.PHONY: help setup models test test-builtin eval demo lint audit llm up up-obs up-all down logs

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:[^#]*## /  -  /'

setup:        ## environment sanity check
	$(PY) scripts/check_setup.py
models:       ## download the local models (LLM, injection classifier, embedder, reranker)
	scripts/setup_models.sh
test:         ## full suite, offline, default (real) embedder and reranker
	$(OFFLINE) $(PY) -m pytest -q
test-builtin: ## full suite on the builtin backend (what the first CI job runs)
	$(OFFLINE) EMBEDDER=builtin RERANKER=builtin $(PY) -m pytest -q
eval:         ## strict scorecard, writes reports/eval_report.md
	$(OFFLINE) $(PY) eval/run_eval.py --strict --real --ablations
demo:         ## the offline demo question
	$(OFFLINE) $(PY) run_assessor.py "Can support use LedgerLens AI with customer tickets?" --tier T3 --users 35
lint:         ## ruff (correctness rules)
	ruff check .
audit:        ## verify the hash-chained audit log
	$(PY) scripts/verify_audit.py
llm:          ## serve the local small model with llama.cpp on :8080 (foreground)
	scripts/serve_local_llm.sh
up:           ## api + ui + sandboxed fetcher (Docker, unverified)
	docker compose up -d --build
up-obs:       ## ... plus Prometheus, Grafana, Loki, Jaeger (Docker, unverified)
	docker compose --profile obs up -d --build
up-all:       ## ... plus the local LLM container (Docker + GPU, unverified)
	docker compose --profile llm --profile obs up -d --build
down:         ## stop the stack
	docker compose --profile llm --profile obs down
logs:         ## follow the API container logs
	docker compose logs -f --tail=100 api
