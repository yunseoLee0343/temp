#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== NVIDIA H100 / sm90 ==="
nvidia-smi || true
python tools/dump_pipeline.py   --sm 90   --gpu-name H100   --stages 1 2 3 4   --out results
python tools/analyze_pipeline_retention.py --root results/H100-sm90 --kernel mini_chunk_fla_pipelined --stages 1 2 3 4
