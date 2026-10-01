#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== NVIDIA L4 / sm89 ==="
nvidia-smi || true
python tools/dump_pipeline.py   --sm 89   --gpu-name L4   --stages 1 2 3 4   --out results
python tools/analyze_pipeline_retention.py --root results/L4-sm89 --kernel mini_chunk_fla_pipelined --stages 1 2 3 4
