#!/usr/bin/env bash
# Serve the local "small" model with llama.cpp on the GPU, as an OpenAI-compatible endpoint on :8080.
#
#   scripts/serve_local_llm.sh                 # foreground
#   LLAMA_CTX=4096 scripts/serve_local_llm.sh  # smaller context if the GPU runs out of memory
#
# Sized for a 4 GB GPU (GTX 1650): a ~4B model at Q4_K_M (2.5 GB) with every layer on the GPU, an 8-bit KV
# cache, and two request slots sharing the context. The app reaches it with LLM_SMALL_PROVIDER=llamacpp and
# falls back to the cloud small model if this server is down (assessor/llm.py).
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${LLAMA_MODEL_FILE:-models/llm/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
PORT="${LLAMA_PORT:-8080}"
HOST="${LLAMA_HOST:-127.0.0.1}"           # loopback only: this server has no user accounts
CTX="${LLAMA_CTX:-8192}"                  # total, shared by the slots
SLOTS="${LLAMA_SLOTS:-2}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"      # 99 = all layers; lower it (or 0 for CPU) if VRAM is short
API_KEY="${LLAMACPP_API_KEY:-local}"

[[ -x "$LLAMA_SERVER" ]] || { echo "llama-server not found at $LLAMA_SERVER (set LLAMA_SERVER)" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "Model file $MODEL is missing. Run scripts/setup_models.sh" >&2; exit 1; }

exec "$LLAMA_SERVER" \
  --model "$MODEL" --alias local-small \
  --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
  --n-gpu-layers "$GPU_LAYERS" --ctx-size "$CTX" --parallel "$SLOTS" \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --reasoning off --metrics --no-webui --timeout 120
