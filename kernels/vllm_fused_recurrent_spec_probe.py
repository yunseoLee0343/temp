# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# Derived from vllm-project/vllm:
#   vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
# upstream blob:
#   eb08b938c2fbc9609a4b5c1ec31477e4f2388a0e
#
# The vLLM file states that portions originate from flash-linear-attention
# under the MIT license, Copyright (c) 2023-2025 Songlin Yang, Yu Zhang.
#
# This probe keeps the production speculative-state indexing semantics:
#   - load the starting recurrent state from
#       ssm_state_indices[num_accepted_tokens - 1]
#   - after every recurrent token, store H_t into
#       ssm_state_indices[i_t]
# Autotune/heuristic wrappers are removed so the specialization is explicit.

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "T"])
def vllm_fused_recurrent_spec_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all_tokens = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_tokens = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv

    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all_tokens + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                # Production rollback / acceptance selection rule.
                selected_t = (
                    tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                )
            else:
                selected_t = 0

            state_idx = tl.load(
                ssm_state_indices
                + i_n * stride_indices_seq
                + selected_t * stride_indices_tok
            ).to(tl.int64)

            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K

        p_h0 = (
            p_h0
            + i_hv * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

        b_q = b_q * scale

        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= tl.exp(b_gk[None, :])

        b_v -= tl.sum(b_h * b_k[None, :], 1)

        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)

        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]

        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Production multi-version recurrent-state materialization:
        # H_{t+1} is stored in the physical slot selected for token t.
        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices
                + i_n * stride_indices_seq
                + i_t * stride_indices_tok
            ).to(tl.int64)

            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = (
                    p_ht
                    + i_hv * V * K
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(
                    p_ht,
                    b_h.to(p_ht.dtype.element_ty),
                    mask=mask_h,
                )
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = (
                p_ht
                + i_hv * V * K
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(
                p_ht,
                b_h.to(p_ht.dtype.element_ty),
                mask=mask_h,
            )

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V

        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K

        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
