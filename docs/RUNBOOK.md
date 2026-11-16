# Runbook

Operating the assessor on one machine. Everything under "Docker" is untested: Docker has not been installed or used on the
development machine, so run `docker compose config` and a first `make up` as a check, not as a routine.

All commands run from the project root with `source .venv/bin/activate`.

## Start and stop (native, the verified path)

```bash
scripts/setup_models.sh                        # once: local LLM, injection classifier, embedder, reranker
scripts/serve_local_llm.sh &                   # local small model on 127.0.0.1:8080 (needs about 3.1 GB of the 4 GB GPU)
VRA_ENV=prod VRA_TOKENS_FILE=runs/tokens.json uvicorn app.api:app --port 8000
streamlit run app/ui.py                        # reviewer UI
```

Stop with Ctrl-C; free the GPU with `pkill -x llama-server`. Do not use `pkill -f` or `pgrep -f` with a pattern that also appears in your
own command line: it will kill your shell.

Quick checks: `curl localhost:8000/health` (liveness), `curl localhost:8000/ready` (database, policy index, model endpoints),
`nvidia-smi` (the model should hold about 3.1 GB).

## Running with Docker (unverified)

```bash
make up          # api + ui + sandboxed fetcher
make up-obs      # + Prometheus :9090, Grafana :3000, Loki, Jaeger :16686
make up-all      # + the llama.cpp container (needs the NVIDIA container toolkit)
make down
```

First things to check: `docker compose ps` (all healthy), `docker compose logs api`, and that the `fetcher` container has no API keys in its
environment and refuses `http://169.254.169.254/` and `http://localhost:8000/`. Until you have seen that, do not rely on the sandbox.

## Tokens and approvers

```bash
python scripts/manage_tokens.py add --user sec.lead@yourco.example --groups employees,security --roles "Head of Information Security"
python scripts/manage_tokens.py add --user dpo@yourco.example       --groups employees,security --roles "DPO"
python scripts/manage_tokens.py list
python scripts/manage_tokens.py revoke --user dpo@yourco.example
```

The script writes `runs/tokens.json` by default, but the API only reads a token file when `VRA_TOKENS_FILE` is set (set it to the same path),
so with it unset no token will ever authenticate. A token is shown once; only its SHA-256 hash is stored. T3 needs two different people holding "Head of Information Security" and "DPO".
**Rotate** by adding a new token for the user, switching them over, then revoking the old entry. Restart the API to reload the file.
Tokens do not expire; revoke when someone leaves.

## Keys and settings

Provider keys live in `.env` (gitignored; never paste it into chat or a ticket). To rotate the MiniMax or Tavily key: replace it in `.env`, restart the
API, run one live assessment. All settings are documented in `.env.example`. In `VRA_ENV=prod`, keep `VRA_MODEL_DOWNLOAD=0` once the models are
cached, so a request can never trigger a download.

## Audit log

```bash
python scripts/verify_audit.py             # exit 0 = intact, 1 = tampered
python scripts/verify_audit.py --tail 20   # and show the last 20 records
```

Run it daily and before relying on the log in a review. A failure means a row was edited or deleted: stop, copy `runs/assessments.db`
somewhere safe before anything else touches it, and find out who had write access.

## Caches

```bash
python scripts/cache_admin.py stats
python scripts/cache_admin.py purge --expired
python scripts/cache_admin.py purge --vendor "Zoho Desk"      # after a vendor changes its terms
python scripts/cache_admin.py purge --ns semantic             # after editing a policy by hand
python scripts/cache_admin.py purge --all --yes
```

Every purge is audited. A re-review that must not use cached pages or searches: `--refresh` on the CLI, `"refresh": true` on the API.

## Retention and backup

```bash
python scripts/retention.py                 # dry run
python scripts/retention.py --apply         # delete run folders > 90 days, rotated logs > 30 days, expired cache; audited
```

It never touches the `assessments` table, the audit log, vendor memory, or runs still waiting for approval. Back up by copying
`runs/assessments.db` (reports, approvals, audit chain, usage, eval history) together with `runs/vendor_memory.json`:

```bash
python - <<'PY'
import sqlite3; s = sqlite3.connect("runs/assessments.db"); d = sqlite3.connect("backup-assessments.db"); s.backup(d)
PY
```

Use the SQLite backup API as above (not `cp`) while the API is running, because the database is in WAL mode. Test a restore by running
`python scripts/verify_audit.py --db backup-assessments.db`.

## Cost control

`VRA_MAX_USD` caps one assessment. `VRA_DAILY_USD` and `VRA_MONTHLY_USD` (0 = off) refuse new live runs once exceeded. The Streamlit
"Cost" and "Operations" tabs and `GET /costs` show spend by day, model and stage. The caps themselves have no automated test; check them once after
setting them.

## Common failures

| Symptom | Cause and fix |
|---|---|
| Tests or `/ready` hang at start | Models are missing and a download is stalling. The loader now gives up after `VRA_MODEL_LOAD_TIMEOUT_S` and uses the builtin backend; fix properly with `HF_HUB_DISABLE_XET=1 scripts/setup_models.sh`. |
| Models re-download after every WSL restart | They were in `/tmp`. The cache is now `~/.cache/fastembed`; check `FASTEMBED_CACHE_PATH`. |
| Live run over 90 s, few facts | Check `runs/<run>/trace.jsonl` for `search_failed` and `fetch_failed`, and look for another transfer using the network. Try `SEARCH_PROVIDER=tavily` with a key. The run stops at `VRA_MAX_SECONDS` and reports what it has. |
| Many `search_failed` with ddgs | DuckDuckGo is rate-limiting or slow from this network. Use Tavily; the other provider is tried automatically if both are configured. |
| `fetch_failed ... pypdf is not installed` | `uv pip install pypdf` (it is in `requirements-prod.txt`). |
| Local model errors, then slower runs | `llama-server` is down or out of GPU memory. `LLM_FALLBACK=1` sends small-role calls to the cloud small model; restart the server (`LLAMA_CTX=4096 scripts/serve_local_llm.sh` if memory is tight). |
| Tests hang or call the live API | `eval/run_eval.py` and `run_assessor.py` read `.env`. Export `LLM_PROVIDER=mock SEARCH_PROVIDER=offline` first (the pytest suite sets it itself). |
| 401 on every call | `VRA_ENV=prod` needs a token on every endpoint except `/health`. |
| 409 on approve | The assessment is BLOCKED; it can only be rejected. |
| 429 | Per-caller rate limit (`VRA_RATE_PER_MIN`). |
