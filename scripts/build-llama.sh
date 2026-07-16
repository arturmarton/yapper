#!/usr/bin/env bash
# Build llama.cpp with CUDA, into wherever .env says llama-server lives.
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

: "${LLAMA_SERVER:?LLAMA_SERVER is not set. cp .env.example .env}"

# LLAMA_SERVER points at <repo>/build/bin/llama-server; walk back to the repo.
REPO="$(cd "$(dirname "$LLAMA_SERVER")/../.." 2>/dev/null && pwd)" \
  || REPO="$(dirname "$(dirname "$(dirname "$LLAMA_SERVER")")")"

if [ ! -d "$REPO" ]; then
  echo "cloning llama.cpp into $REPO"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$REPO"
fi

cd "$REPO"
# 89 = RTX 4090 (sm_89). Change this for a different card.
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF
cmake --build build --config Release -j "$(nproc)"

echo "built: $LLAMA_SERVER"
