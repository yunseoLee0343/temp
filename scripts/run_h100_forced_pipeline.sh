#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== H100 source-driven software-pipeline probe / sm90 ==="
nvidia-smi || true

for STAGE in 1 2 3 4; do
  DUMP="results/H100-sm90/pass-dumps/mini_chunk_fla_forced.stage-${STAGE}.mlir.txt"
  echo
  echo "=== stage ${STAGE} ==="
  rm -f "$DUMP"
  python tools/dump_forced_passes.py \
    --sm 90 \
    --gpu-name H100 \
    --stage "$STAGE" \
    --out results \
    --pass-dump "$DUMP"
done

python tools/analyze_pipeline_retention.py \
  --root results/H100-sm90 \
  --kernel mini_chunk_fla_forced \
  --stages 1 2 3 4

python tools/analyze_pass_dumps.py \
  --dump-dir results/H100-sm90/pass-dumps \
  --out results/H100-sm90/analysis/mini_chunk_fla_forced_pass_dumps.json

echo
echo "Forced-pipeline artifacts:"
echo "  results/H100-sm90/stages-{1,2,3,4}/mini_chunk_fla_forced/"
echo "Pass-manager dumps:"
echo "  results/H100-sm90/pass-dumps/"
