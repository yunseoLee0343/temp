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
