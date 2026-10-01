import triton
import triton.language as tl


@triton.jit
def mini_recurrent(
    q,
    k,
    v,
    g,
    beta,
    h0,
    out,
    ht,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
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

    q_ptr = q + offs_k
    k_ptr = k + offs_k
    v_ptr = v + offs_v
    g_ptr = g
    beta_ptr = beta

    for t in range(0, T):
        q_t = tl.load(q_ptr, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_ptr, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_ptr, mask=mask_v, other=0.0).to(tl.float32)

        decay = tl.exp(tl.load(g_ptr))
        b = tl.load(beta_ptr)

        h_decay = h * decay
        proj = tl.sum(h_decay * k_t[None, :], axis=1)
        v_new = (v_t - proj) * b
        h = h_decay + v_new[:, None] * k_t[None, :]
        o_t = tl.sum(h * q_t[None, :], axis=1)

        tl.store(out + t * V + offs_v, o_t, mask=mask_v)

        q_ptr += K
        k_ptr += K
        v_ptr += V
        g_ptr += 1
        beta_ptr += 1

    ht_ptrs = ht + offs_v[:, None] * K + offs_k[None, :]
    tl.store(ht_ptrs, h, mask=mask_h)
