import triton
import triton.language as tl


@triton.jit
def mini_chunk_fla(
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
):
    pid_v = tl.program_id(0)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)

    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    h_ptrs = h0 + offs_v[:, None] * K + offs_k[None, :]
    h = tl.load(h_ptrs, mask=mask_h, other=0.0).to(tl.float32)

    num_chunks = tl.cdiv(T, BT)

    for c in range(0, num_chunks):
        offs_t = c * BT + tl.arange(0, BT)
        mask_t = offs_t < T

        k_ptrs = (
            k
            + offs_t[:, None] * K
            + offs_k[None, :]
        )

        v_ptrs = (
            v
            + offs_t[:, None] * V
            + offs_v[None, :]
        )

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

        w_ptrs = (
            w
            + offs_t[:, None] * K
            + offs_k[None, :]
        )

        w_tile = tl.load(
            w_ptrs,
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        )

        correction = tl.dot(
            w_tile,
            tl.trans(h).to(w_tile.dtype),
        )

        v_new = v_tile - correction

        delta_h = tl.dot(
            tl.trans(k_tile),
            v_new,
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
