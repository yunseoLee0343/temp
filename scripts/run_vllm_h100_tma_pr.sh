#!/usr/bin/env bash
# H100 SM90; Lightning cloudspace Python 3.14 + torch cu128 + Triton 3.6.
# No new environment, pip install or vLLM dependency.
set -Eeuo pipefail
BRANCH="experiment/vllm-unified-attn-td-probe"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
test "$(git branch --show-current)" = "$BRANCH" || { echo "Checkout $BRANCH first"; exit 2; }
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$ROOT/results/H100-sm90/vllm-unified-attention-td-probe/$RUN_ID"
mkdir -p "$DEST/kernel-dump" "$DEST/pass-dumps" "$DEST/logs"
export RESULT_DIR="$DEST" TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 MLIR_ENABLE_DUMP=1
export TRITON_DUMP_DIR="$DEST/kernel-dump" MLIR_DUMP_PATH="$DEST/pass-dumps/combined-pass-dump.mlir.txt"
export TRITON_CACHE_DIR="$DEST/triton-cache"
python - <<'PY' | tee "$DEST/environment.txt"
import torch, triton, sys, subprocess
print("python",sys.version)
print("torch",torch.__version__,"cuda",torch.version.cuda,"triton",triton.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("gpu",torch.cuda.get_device_name(0),"capability",torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0)==(9,0), "Requires H100 SM90"
print("git",subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
PY
cat > "$DEST/td_probe.py" <<'PY'
"""Isolated vLLM-inspired TensorDescriptor control, not upstream vLLM."""
import os, json
from pathlib import Path
import torch, triton
import triton.language as tl

@triton.jit
def attention_td_control(X, BLOCK_TABLE, OUT,
                         M:tl.constexpr,N:tl.constexpr,
                         BM:tl.constexpr,BN:tl.constexpr):
    desc=tl.make_tensor_descriptor(X,shape=[M,N],strides=[N,1],
                                   block_shape=[BM,BN])
    pid=tl.program_id(0)
    for t in range(0,2):
        block=tl.load(BLOCK_TABLE+pid*2+t)
        v=desc.load([block*BM,0])
        reduced=tl.sum(v,axis=1)
        rows=(pid*2+t)*BM+tl.arange(0,BM)
        tl.store(OUT+rows,reduced)

def main():
    triton.set_allocator(lambda size,alignment,stream:
                         torch.empty(size,device="cuda",dtype=torch.int8))
    BM,BN,M,N=32,128,256,128
    x=torch.arange(M*N,device="cuda",dtype=torch.float32).reshape(M,N)/1024
    block_table=torch.arange(M//BM-1,-1,-1,device="cuda",dtype=torch.int32)
    out=torch.empty(M,device="cuda",dtype=torch.float32)
    kernel=attention_td_control[(M//(2*BM),)](
        x,block_table,out,M,N,BM,BN,num_warps=4)
    torch.cuda.synchronize()
    ref=x.reshape(M//BM,BM,N)[block_table.long()].sum(-1).reshape(-1)
    torch.testing.assert_close(out,ref,rtol=1e-4,atol=1e-3)
    dest=Path(os.environ["RESULT_DIR"])
    info={"correctness":"PASS","max_abs_error":(out-ref).abs().max().item(),
          "triton":triton.__version__,"torch":torch.__version__}
    (dest/"probe_result.json").write_text(json.dumps(info,indent=2)+"\n")
    print(info)
    for fmt in ("ttir","ttgir","llir","ptx"):
        data=kernel.asm.get(fmt)
        if data: (dest/("attention_td_control."+fmt)).write_text(str(data))
    cubin=kernel.asm.get("cubin")
    if isinstance(cubin,bytes):
        (dest/"attention_td_control.cubin").write_bytes(cubin)
if __name__=="__main__": main()
PY
set +e
python "$DEST/td_probe.py" > "$DEST/logs/probe.stdout.log" 2> "$DEST/logs/probe.stderr.log"
RC=$?
set -e
echo "$RC" > "$DEST/logs/exit-code.txt"
if [[ ! -s "$MLIR_DUMP_PATH" ]] && grep -q 'IR Dump Before' "$DEST/logs/probe.stderr.log"; then
  cp "$DEST/logs/probe.stderr.log" "$MLIR_DUMP_PATH"
fi
python - "$DEST" "$RC" <<'PY'
from pathlib import Path
import json,sys
d=Path(sys.argv[1]); rc=int(sys.argv[2])
files=list(d.rglob("*.ptx"))
hits=[]
for p in files:
    for s in p.read_text(errors="replace").splitlines():
        if "cp.async.bulk.tensor" in s:
            hits.append({"ptx":str(p.relative_to(d)),"line":s.strip()[:160]})
status="TMA_PTX_CONFIRMED" if rc==0 and hits else ("PROBE_FAILED" if rc else "NO_TMA_PTX_EVIDENCE")
obj={"status":status,"exit_code":rc,"ptx_count":len(files),
     "tensor_tma_count":len(hits),"samples":hits[:20],
     "provenance":"standalone vLLM-inspired TD control, not full vLLM"}
(d/"evidence.json").write_text(json.dumps(obj,indent=2)+"\n")
print(json.dumps(obj,indent=2))
PY
cat > "$DEST/README.md" <<EOF
# Standalone H100 descriptor/TMA control: $RUN_ID
This probe is **not an execution of vLLM upstream attention**.
It follows dynamic block-table -> tensor descriptor load -> reduction.
Check evidence.json for PTX \`cp.async.bulk.tensor\` and probe_result.json for correctness.
The pass dump is a pre-pass snapshot sequence, not automatically a LICM bug.
EOF
# Keep GitHub commits bounded; preserve excluded files locally.
find "$DEST" -type f -size +40M -print > "$DEST/logs/skipped-large-files.txt"
# Avoid adding oversized files; record small files only.
while IFS= read -r f; do
  if ! grep -Fxq "$f" "$DEST/logs/skipped-large-files.txt"; then
    git add -f -- "$f"
  fi
done < <(find "$DEST" -type f)
if ! git diff --cached --quiet; then
  git -c user.name="H100 Experiment" -c user.email="h100-experiment@users.noreply.github.com" \
    commit -m "experiment: SM90 descriptor probe $RUN_ID"
fi
echo "[artifacts] $DEST"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  git push -u origin "$BRANCH"
  if ! gh pr view "$BRANCH" --json url >/dev/null 2>&1; then
    gh pr create --draft --base main --head "$BRANCH" \
      --title "Experiment: SM90 descriptor/TMA control" \
      --body "Standalone vLLM-inspired TD control (not vLLM upstream). See evidence.json and compiler dumps."
  fi
else
  echo "No authenticated gh CLI. Local commit retained."
  echo "Authenticate, then: git push -u origin $BRANCH"
fi
if (( RC != 0 )); then
  echo "Probe failed; inspect $DEST/logs/probe.stderr.log" >&2
  exit "$RC"
fi
