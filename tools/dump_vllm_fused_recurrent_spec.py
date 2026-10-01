#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.vllm_fused_recurrent_spec_probe import (
    vllm_fused_recurrent_spec_kernel,
)


def write_artifact(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (bytes, bytearray)):
        path.write_bytes(bytes(value))
    else:
        path.write_text(str(value), encoding="utf-8")


def maybe_disassemble(cubin: Path, sass: Path):
    nvdisasm = shutil.which("nvdisasm")
    if nvdisasm:
        proc = subprocess.run([nvdisasm, str(cubin)], text=True, capture_output=True)
        if proc.returncode == 0:
            sass.write_text(proc.stdout, encoding="utf-8")
            return {"tool": "nvdisasm", "ok": True}
        return {"tool": "nvdisasm", "ok": False, "stderr": proc.stderr}

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump:
        proc = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubin)],
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            sass.write_text(proc.stdout, encoding="utf-8")
            return {"tool": "cuobjdump", "ok": True}
        return {"tool": "cuobjdump", "ok": False, "stderr": proc.stderr}

    return {"tool": None, "ok": False, "stderr": "nvdisasm/cuobjdump not found"}


def source_spec_recurrent():
    signature = {
        "q": "*fp32",
        "k": "*fp32",
        "v": "*fp32",
        "g": "*fp32",
        "beta": "*fp32",
        "o": "*fp32",
        "h0": "*fp32",
        "ht": "*fp32",
        "cu_seqlens": "*i32",
        "ssm_state_indices": "*i32",
        "num_accepted_tokens": "*i32",
        "scale": "fp32",
        "N": "i64",
        "T": "i64",
        "B": "constexpr",
        "H": "constexpr",
        "HV": "constexpr",
        "K": "constexpr",
        "V": "constexpr",
        "BK": "constexpr",
        "BV": "constexpr",
        "stride_init_state_token": "constexpr",
        "stride_final_state_token": "constexpr",
        "stride_indices_seq": "constexpr",
        "stride_indices_tok": "constexpr",
        "USE_INITIAL_STATE": "constexpr",
        "INPLACE_FINAL_STATE": "constexpr",
        "IS_BETA_HEADWISE": "constexpr",
        "USE_QK_L2NORM_IN_KERNEL": "constexpr",
        "IS_VARLEN": "constexpr",
        "IS_CONTINUOUS_BATCHING": "constexpr",
        "IS_SPEC_DECODING": "constexpr",
        "IS_KDA": "constexpr",
    }
    constants = {
        "B": 1,
        "H": 1,
        "HV": 1,
        "K": 32,
        "V": 32,
        "BK": 32,
        "BV": 32,
        "stride_init_state_token": 32 * 32,
        "stride_final_state_token": 32 * 32,
        "stride_indices_seq": 5,
        "stride_indices_tok": 1,
        "USE_INITIAL_STATE": True,
        "INPLACE_FINAL_STATE": True,
        "IS_BETA_HEADWISE": False,
        "USE_QK_L2NORM_IN_KERNEL": False,
        "IS_VARLEN": False,
        "IS_CONTINUOUS_BATCHING": True,
        "IS_SPEC_DECODING": True,
        "IS_KDA": False,
    }
    return ASTSource(
        fn=vllm_fused_recurrent_spec_kernel,
        signature=signature,
        constexprs=constants,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", type=int, default=90)
    ap.add_argument("--gpu-name", default="H100")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

    target = GPUTarget("cuda", args.sm, 32)
    root = args.out / f"{args.gpu_name}-sm{args.sm}" / "vllm-fused-recurrent"
    out_dir = root / "compile"
    out_dir.mkdir(parents=True, exist_ok=True)

    compiled = triton.compile(
        source_spec_recurrent(),
        target=target,
        options={"num_warps": 1, "num_stages": 3},
    )

    ext_map = {
        "ttir": ".ttir",
        "ttgir": ".ttgir",
        "llir": ".llir",
        "ptx": ".ptx",
        "cubin": ".cubin",
    }
    saved = []
    for key, ext in ext_map.items():
        if key in compiled.asm:
            write_artifact(out_dir / f"vllm_fused_recurrent_spec{ext}", compiled.asm[key])
            saved.append(key)

    disasm = None
    cubin = out_dir / "vllm_fused_recurrent_spec.cubin"
    if cubin.exists():
        disasm = maybe_disassemble(
            cubin,
            out_dir / "vllm_fused_recurrent_spec.sass",
        )

    metadata = {
        "gpu_label": args.gpu_name,
        "sm": args.sm,
        "triton_version": triton.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "num_warps": 1,
        "num_stages": 3,
        "artifact_keys": sorted(compiled.asm.keys()),
        "saved_artifacts": saved,
        "disassembly": disasm,
        "upstream_source": {
            "path": "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
            "blob": "eb08b938c2fbc9609a4b5c1ec31477e4f2388a0e",
        },
        "probe_specialization": {
            "B": 1,
            "H": 1,
            "HV": 1,
            "K": 32,
            "V": 32,
            "BK": 32,
            "BV": 32,
            "IS_CONTINUOUS_BATCHING": True,
            "IS_SPEC_DECODING": True,
            "INPLACE_FINAL_STATE": True,
            "stride_indices_seq": 5,
        },
        "note": (
            "T remains runtime/dynamic, matching upstream do_not_specialize. "
            "Runtime num_spec experiments are recorded separately."
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
