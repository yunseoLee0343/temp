#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== vLLM FLA fixed-stage software-pipelining experiment / H100 sm90 ==="
nvidia-smi || true

BASE="results/H100-sm90/vllm-fla"
DUMP_DIR="$BASE/pass-dumps"
mkdir -p "$DUMP_DIR"

for STAGE in 2 3 4; do
  echo
  echo "=== positive control: chunk_gla_fwd_kernel_o / stage ${STAGE} ==="
  POS_DUMP="$DUMP_DIR/vllm_chunk_gla_o.stage-${STAGE}.mlir.txt"
  rm -f "$POS_DUMP"
  TRITON_ALWAYS_COMPILE=1 \
  MLIR_ENABLE_DUMP=1 \
  MLIR_DUMP_PATH="$POS_DUMP" \
    python tools/dump_vllm_fla_pipeline.py \
      --sm 90 \
      --gpu-name H100 \
      --stage "$STAGE" \
      --kernel positive \
      --out results

  echo
  echo "=== recurrent target: chunk_delta_h / stage ${STAGE} ==="
  REC_DUMP="$DUMP_DIR/vllm_chunk_delta_h.stage-${STAGE}.mlir.txt"
  rm -f "$REC_DUMP"
  TRITON_ALWAYS_COMPILE=1 \
  MLIR_ENABLE_DUMP=1 \
  MLIR_DUMP_PATH="$REC_DUMP" \
    python tools/dump_vllm_fla_pipeline.py \
      --sm 90 \
      --gpu-name H100 \
      --stage "$STAGE" \
      --kernel recurrent \
      --out results
done

python tools/analyze_pipeline_retention.py \
  --root "$BASE" \
  --kernel vllm_chunk_gla_o \
  --stages 2 3 4

python tools/analyze_pipeline_retention.py \
  --root "$BASE" \
  --kernel vllm_chunk_delta_h \
  --stages 2 3 4

python tools/analyze_pass_dumps.py \
  --dump-dir "$DUMP_DIR" \
  --glob "vllm_chunk_gla_o.stage-*.mlir.txt" \
  --out "$BASE/analysis/vllm_chunk_gla_o_pass_dumps.json"

python tools/analyze_pass_dumps.py \
  --dump-dir "$DUMP_DIR" \
  --glob "vllm_chunk_delta_h.stage-*.mlir.txt" \
  --out "$BASE/analysis/vllm_chunk_delta_h_pass_dumps.json"

echo
echo "Done."
echo "Final artifacts: $BASE/stages-{2,3,4}/"
echo "Raw MLIR pass dumps: $DUMP_DIR/"
echo "Analysis: $BASE/analysis/"
