import triton
import triton.language as tl


@triton.jit
def mini_chunk_fla_pipelined(
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
    Same recurrence as mini_chunk_fla, but the chunk loop carries an explicit
    tl.range(..., num_stages=PIPE_STAGES) hint.

    Purpose: force Triton's loop pipeliner to consider future-iteration K/V/W
    loads independently from the loop-carried H state, so stage-dependent
    shared-memory / async-copy expansion can be observed in TTGIR.
    """
    pid_v = tl.program_id(0)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)

    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    h_ptrs = h0 + offs_v[:, None] * K + offs_k[None, :]
    h = tl.load(h_ptrs, mask=mask_h, other=0.0).to(tl.float32)

    num_chunks = tl.cdiv(T, BT)

    for c in tl.range(0, num_chunks, num_stages=PIPE_STAGES):
        offs_t = c * BT + tl.arange(0, BT)
        mask_t = offs_t < T

        k_ptrs = k + offs_t[:, None] * K + offs_k[None, :]
        v_ptrs = v + offs_t[:, None] * V + offs_v[None, :]
        w_ptrs = w + offs_t[:, None] * K + offs_k[None, :]

        # These three loads are loop-iteration-local and do not depend on H_c.
        # They are the intended software-pipeline candidates.
        k_tile = tl.load(
            k_ptrs,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        )
        v_tile = tl.load(
            v_ptrs,
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        )

        # H_c remains a true loop-carried recurrence.
        correction = tl.dot(
            w_tile,
            tl.trans(h).to(w_tile.dtype),
        )

        v_new = v_tile - correction
        v_new_dot = v_new.to(k_tile.dtype)

        delta_h = tl.dot(
            tl.trans(k_tile),
            v_new_dot,
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
