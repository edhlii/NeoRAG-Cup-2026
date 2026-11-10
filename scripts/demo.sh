#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo 'Thiếu .venv. Xem docs/FINAL.md để cài đặt.' >&2
  exit 1
fi
docker compose up -d
export NEORAG_PIPELINE=neorag.pipelines.hybrid:HybridPipeline
export NEORAG_HYBRID_CONFIG=configs/hybrid.toml
exec .venv/bin/python -m streamlit run app/chat.py "$@"
