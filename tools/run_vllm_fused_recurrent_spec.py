#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.vllm_fused_recurrent_spec_probe import (
    vllm_fused_recurrent_spec_kernel,
)


H = 1
HV = 1
K = 32
V = 32
BK = 32
BV = 32
STATE_ELEMS = HV * V * K


def make_inputs(T: int, device: torch.device, phase: float):
    # Small deterministic fp32 values keep the reference numerically stable.
    t = torch.arange(T, device=device, dtype=torch.float32)
    kk = torch.arange(K, device=device, dtype=torch.float32)
    vv = torch.arange(V, device=device, dtype=torch.float32)

    q = (
        0.01
        + 0.0007 * kk[None, None, None, :]
        + 0.0003 * t[None, :, None, None]
        + phase
    ).contiguous()
    k = (
        0.02
        + 0.0005 * kk[None, None, None, :]
        + 0.0002 * t[None, :, None, None]
        + phase * 0.5
    ).contiguous()
    v = (
        0.03
        + 0.0004 * vv[None, None, None, :]
        + 0.00025 * t[None, :, None, None]
        + phase * 0.25
    ).contiguous()
    g = (-0.04 - 0.003 * t[None, :, None]).contiguous()
    beta = (0.55 + 0.01 * t[None, :, None]).contiguous()
    return q, k, v, g, beta


def reference_states(
    initial_h: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
):
    h = initial_h.clone().to(torch.float32)
    states = []
    T = k.shape[1]
    for i in range(T):
        k_t = k[0, i, 0].to(torch.float32)
        v_t = v[0, i, 0].to(torch.float32)
        g_t = g[0, i, 0].to(torch.float32)
        beta_t = beta[0, i, 0].to(torch.float32)

        h = h * torch.exp(g_t)
        v_new = v_t - torch.sum(h * k_t[None, :], dim=1)
        v_new = v_new * beta_t
        h = h + v_new[:, None] * k_t[None, :]
        states.append(h.clone())
    return states


def launch(
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    accepted: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
):
    T = int(k.shape[1])
    o = torch.empty((1, T, HV, V), device=k.device, dtype=torch.float32)
    cu = torch.tensor([0, T], device=k.device, dtype=torch.int32)
    num_accepted = torch.tensor([accepted], device=k.device, dtype=torch.int32)

    vllm_fused_recurrent_spec_kernel[(1, 1, 1)](
        q,
        k,
        v,
        g,
        beta,
        o,
        state_cache,
        state_cache,
        cu,
        state_indices,
        num_accepted,
        1.0,
        1,
        T,
        B=1,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=STATE_ELEMS,
        stride_final_state_token=STATE_ELEMS,
        stride_indices_seq=state_indices.stride(0),
        stride_indices_tok=state_indices.stride(1),
        USE_INITIAL_STATE=True,
        INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=False,
        USE_QK_L2NORM_IN_KERNEL=False,
        IS_VARLEN=False,
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=True,
        IS_KDA=False,
        num_warps=1,
        num_stages=3,
    )
    torch.cuda.synchronize()
    return o


def tensor_summary(x: torch.Tensor):
    flat = x.detach().float().flatten()
    return {
        "sum": float(flat.sum().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "max_abs": float(flat.abs().max().item()),
        "sample": [float(v) for v in flat[:8].cpu().tolist()],
    }


def run_case(num_spec: int, out_dir: Path, device: torch.device):
    versions = num_spec + 1

    # Slot 0 is reserved as NULL_BLOCK_ID. Slots 1..versions are the physical
    # recurrent-state family selected by ssm_state_indices.
    state_cache = torch.zeros(
        (versions + 1, HV, V, K),
        device=device,
        dtype=torch.float32,
    )
    state_indices = torch.arange(
        1,
        versions + 1,
        device=device,
        dtype=torch.int32,
    )[None, :].contiguous()

    q, k, v, g, beta = make_inputs(versions, device, phase=0.0)
    initial_h = state_cache[1, 0].clone()
    refs = reference_states(initial_h, k, v, g, beta)

    # accepted=1 selects state_indices[0] as H0. The loop then writes H1..HT
    # into slots state_indices[0..T-1].
    launch(
        state_cache,
        state_indices,
        accepted=1,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
    )

    candidate_snapshot = state_cache.clone()
    candidate_errors = []
    slots = []
    for i, ref in enumerate(refs):
        slot_id = int(state_indices[0, i].item())
        got = candidate_snapshot[slot_id, 0]
        err = float((got - ref).abs().max().item())
        candidate_errors.append(err)
        slots.append(
            {
                "semantic_version": f"H_{i + 1}",
                "token_index": i,
                "physical_slot": slot_id,
                "max_abs_error_vs_reference": err,
                "summary": tensor_summary(got),
            }
        )

    # Verify acceptance-conditioned selection. For each possible accepted
    # prefix a, clone the multi-version snapshot, run one extra recurrent token,
    # and check that the result equals F(H_a, x_extra). The kernel loads
    # state_indices[a-1] before overwriting state_indices[0].
    qx, kx, vx, gx, betax = make_inputs(1, device, phase=0.013)
    acceptance_checks = []
    for accepted in range(1, versions + 1):
        trial_cache = candidate_snapshot.clone()
        selected_slot = int(state_indices[0, accepted - 1].item())
        selected_h = candidate_snapshot[selected_slot, 0].clone()
        expected_next = reference_states(selected_h, kx, vx, gx, betax)[0]

        launch(
            trial_cache,
            state_indices,
            accepted=accepted,
            q=qx,
            k=kx,
            v=vx,
            g=gx,
            beta=betax,
        )

        write_slot = int(state_indices[0, 0].item())
        got_next = trial_cache[write_slot, 0]
        err = float((got_next - expected_next).abs().max().item())
        acceptance_checks.append(
            {
                "num_accepted_tokens": accepted,
                "selected_column": accepted - 1,
                "selected_physical_slot": selected_slot,
                "selected_semantic_state": f"H_{accepted}",
                "next_write_slot": write_slot,
                "max_abs_error_vs_selected_state_reference": err,
            }
        )

    result = {
        "num_spec": num_spec,
        "physical_state_versions": versions,
        "state_indices": state_indices.cpu().tolist(),
        "null_slot": 0,
        "candidate_materialization": {
            "max_error": max(candidate_errors),
            "slots": slots,
        },
        "acceptance_conditioned_reuse": {
            "max_error": max(x["max_abs_error_vs_selected_state_reference"] for x in acceptance_checks),
            "checks": acceptance_checks,
        },
        "kernel_contract": {
            "store_rule": "H_{t+1} -> state_cache[ssm_state_indices[t]]",
            "selection_rule": "H_start <- state_cache[ssm_state_indices[num_accepted_tokens-1]]",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"num-spec-{num_spec}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/H100-sm90/vllm-fused-recurrent/runtime"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    device = torch.device("cuda")
    all_results = []
    for num_spec in [1, 2, 3, 4]:
        print(f"[run] num_spec={num_spec} -> {num_spec + 1} recurrent-state slots")
        r = run_case(num_spec, args.out, device)
        all_results.append(r)
        print(
            "  materialization max error:",
            r["candidate_materialization"]["max_error"],
            "acceptance max error:",
            r["acceptance_conditioned_reuse"]["max_error"],
        )

    summary = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cases": [
            {
                "num_spec": r["num_spec"],
                "physical_state_versions": r["physical_state_versions"],
                "state_indices": r["state_indices"],
                "candidate_max_error": r["candidate_materialization"]["max_error"],
                "acceptance_max_error": r["acceptance_conditioned_reuse"]["max_error"],
            }
            for r in all_results
        ],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "vLLM fused recurrent speculative-state retention",
        "",
        "num_spec | physical H slots | candidate error | acceptance/rollback error",
    ]
    for r in all_results:
        lines.append(
            f"{r['num_spec']:>8} | "
            f"{r['physical_state_versions']:>16} | "
            f"{r['candidate_materialization']['max_error']:.8g} | "
            f"{r['acceptance_conditioned_reuse']['max_error']:.8g}"
        )
    (args.out / "summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
