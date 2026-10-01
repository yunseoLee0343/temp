#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== vLLM fused recurrent speculative-state experiment / H100 sm90 ==="
nvidia-smi || true

BASE="results/H100-sm90/vllm-fused-recurrent"
mkdir -p "$BASE/pass-dumps" "$BASE/runtime"

DUMP="$BASE/pass-dumps/fused_recurrent_spec.mlir.txt"
rm -f "$DUMP"

echo
echo "=== compiler artifacts + MLIR pass dump ==="
TRITON_ALWAYS_COMPILE=1 \
MLIR_ENABLE_DUMP=1 \
MLIR_DUMP_PATH="$DUMP" \
  python tools/dump_vllm_fused_recurrent_spec.py \
    --sm 90 \
    --gpu-name H100 \
    --out results

echo
echo "=== runtime semantic-state materialization / rollback checks ==="
TRITON_ALWAYS_COMPILE=1 \
  python tools/run_vllm_fused_recurrent_spec.py \
    --out "$BASE/runtime"

echo
echo "Summary:"
cat "$BASE/runtime/summary.txt"

echo
echo "Artifacts:"
echo "  $BASE/compile/"
echo "  $BASE/pass-dumps/"
echo "  $BASE/runtime/"
echo
echo "Publish with:"
echo "  bash scripts/commit_results.sh h100"
