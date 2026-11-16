#!/usr/bin/env bash
# Download the local models into ./models (git-ignored). Safe to re-run: finished files are skipped.
#
#   scripts/setup_models.sh            # LLM + prompt-injection classifier
#   scripts/setup_models.sh --no-llm   # classifier only
#
# The embedder (BAAI/bge-small-en-v1.5) and reranker (Xenova/ms-marco-MiniLM-L-6-v2) are fetched by fastembed
# on first use and cached under ~/.cache (or FASTEMBED_CACHE_PATH).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_TELEMETRY=1
HF="${HF:-hf}"
command -v "$HF" >/dev/null || { echo "The 'hf' CLI is missing: pip install huggingface_hub" >&2; exit 1; }

LLM_REPO="${LLM_REPO:-bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF}"
LLM_FILE="${LLM_FILE:-Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
INJ_REPO="${INJ_REPO:-protectai/deberta-v3-base-prompt-injection-v2}"

mkdir -p models/llm models/injection
if [[ "${1:-}" != "--no-llm" ]]; then
  echo "LLM: $LLM_REPO / $LLM_FILE (about 2.5 GB)"
  "$HF" download "$LLM_REPO" "$LLM_FILE" --local-dir models/llm
fi
echo "Prompt-injection classifier: $INJ_REPO (ONNX, CPU)"
"$HF" download "$INJ_REPO" --include "onnx/*" --include "config.json" --include "tokenizer.json" --include "LICENSE" \
  --local-dir "models/injection/$(basename "$INJ_REPO")"
echo
echo "Done. Add to .env:"
echo "  LLM_SMALL_PROVIDER=llamacpp"
echo "  INJECTION_MODEL=models/injection/$(basename "$INJ_REPO")/onnx"
