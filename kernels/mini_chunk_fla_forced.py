import triton
import triton.language as tl


@triton.jit
def mini_chunk_fla_forced(
    k,
    v,
    w,
    h0,
    chunk_state,
    out,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    """
    Source-driven software-pipeline probe derived from Triton's pipeliner gates.

    The steady-state loop intentionally uses only full BT-sized chunks, so the
    K/V/W global loads need no runtime tail mask.  This makes their contiguous
    fp16 width visible to ModuleAxisInfoAnalysis and gives
    canBeConvertedToAsyncLoad() a clean >= 32-bit candidate.

    K/W feed tl.dot directly; V feeds the same dot chain indirectly.  The H
    value remains a genuine distance-1 loop-carried recurrent state.

    Tail elements (T % BT) are intentionally not handled: this kernel is a
    compiler transformation probe and is not launched by the dump harness.
    """
    pid_v = tl.program_id(0)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)

    # The compile harness specializes K=V=BK=BV=16, so these become true.
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    h_ptrs = h0 + offs_v[:, None] * K + offs_k[None, :]
    h = tl.load(h_ptrs, mask=mask_h, other=0.0).to(tl.float32)

    # Only full chunks participate in the pipelined steady state.
    num_full_chunks = T // BT

    for c in tl.range(
        0,
        num_full_chunks,
        num_stages=PIPE_STAGES,
    ):
        offs_t = c * BT + tl.arange(0, BT)

        k_ptrs = k + offs_t[:, None] * K + offs_k[None, :]
        v_ptrs = v + offs_t[:, None] * V + offs_v[None, :]
        w_ptrs = w + offs_t[:, None] * K + offs_k[None, :]

        # No runtime mask / other value here by design.  These are the loads
        # that AssignLoadLatencies -> lowerLoads should try to pipeline.
        k_tile = tl.load(k_ptrs)
        v_tile = tl.load(v_ptrs)
        w_tile = tl.load(w_ptrs)

        # Direct load -> dot path for W. H_c is the recurrent operand.
        correction = tl.dot(
            w_tile,
            tl.trans(h).to(tl.float16),
        )

        v_new = v_tile.to(tl.float32) - correction

        # Direct load -> dot path for K; V reaches this dot through v_new.
        delta_h = tl.dot(
            tl.trans(k_tile),
            v_new.to(tl.float16),
        )

        h = h + tl.trans(delta_h)

        h_out_ptrs = (
            chunk_state
            + c * V * K
            + offs_v[:, None] * K
            + offs_k[None, :]
        )
        tl.store(h_out_ptrs, h, mask=mask_h)

        o = tl.sum(h, axis=1)
        tl.store(out + c * V + offs_v, o, mask=mask_v)
