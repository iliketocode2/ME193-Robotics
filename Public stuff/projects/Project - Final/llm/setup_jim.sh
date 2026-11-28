#!/bin/bash
# Set up the on-board LLM on an Arduino UNO Q (Debian 13, aarch64, 4 GB).
# Run on the board as the arduino user, no sudo needed:
#
#   scp -r llm arduino@<board-ip>:~/final-llm && ssh arduino@<board-ip> bash ~/final-llm/setup_jim.sh
#
# What it does:
#   1. downloads a prebuilt llama.cpp (CPU, linux arm64) into ~/llama/<build>,
#      symlinked as ~/llama/current (the board has no compiler installed)
#   2. downloads the candidate models (GGUF, 4-bit Q4_K_M) into ~/models.
#      Models go under /home because / only has ~3 GB free.
#   3. points ~/models/active.gguf at the chosen model and installs
#      llama-server as a systemd user service that starts at boot
set -euo pipefail

LLAMA_BUILD=${LLAMA_BUILD:-b12823}
ACTIVE=${ACTIVE:-Qwen3-1.7B-Q4_K_M.gguf}
HERE=$(cd "$(dirname "$0")" && pwd)

MODELS=(
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf
  https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
)

# 1. llama.cpp
mkdir -p ~/llama ~/models
if [ ! -x ~/llama/llama-$LLAMA_BUILD/llama-server ]; then
  echo "downloading llama.cpp $LLAMA_BUILD"
  curl -fsSL -o /tmp/llama.tgz \
    "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-bin-ubuntu-arm64.tar.gz"
  tar xzf /tmp/llama.tgz -C ~/llama && rm /tmp/llama.tgz
fi
ln -sfn ~/llama/llama-$LLAMA_BUILD ~/llama/current
~/llama/current/llama-cli --version 2>&1 | head -1

# 2. models
for u in "${MODELS[@]}"; do
  f=~/models/$(basename "$u")
  if [ ! -s "$f" ]; then
    echo "downloading $(basename "$u")"
    curl -fsSL --retry 3 -o "$f.part" "$u" && mv "$f.part" "$f"
  fi
done
ls -lh ~/models/*.gguf

# 3. service
ln -sfn ~/models/$ACTIVE ~/models/active.gguf
mkdir -p ~/.config/systemd/user
cp "$HERE/llama-server.service" ~/.config/systemd/user/
loginctl enable-linger "$USER"          # start user services at boot, no login needed
systemctl --user daemon-reload
systemctl --user enable --now llama-server
echo "llama-server enabled with $ACTIVE on port 8090"
