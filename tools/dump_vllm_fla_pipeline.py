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

from kernels.vllm_fla_pipeline_probes import (
    vllm_chunk_delta_h_kernel,
    vllm_chunk_gla_o_kernel,
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


def source_positive_control():
    # Fixed from one of upstream's valid autotune choices:
    # BK=64, BV=64, num_warps=4. K=128 ensures the i_k loop has two trips.
    signature = {
        "q": "*fp16",
        "v": "*fp16",
        "g": "*fp32",
        "h": "*fp16",
        "o": "*fp16",
        "A": "*fp16",
        "cu_seqlens": "*i32",
        "chunk_indices": "*i32",
        "scale": "fp32",
        "T": "i32",
        "H": "constexpr",
        "K": "constexpr",
        "V": "constexpr",
        "BT": "constexpr",
        "BK": "constexpr",
        "BV": "constexpr",
        "IS_VARLEN": "constexpr",
    }
    constants = {
        "H": 1,
        "K": 128,
        "V": 64,
        "BT": 64,
        "BK": 64,
        "BV": 64,
        "IS_VARLEN": False,
    }
    return ASTSource(
        fn=vllm_chunk_gla_o_kernel,
        signature=signature,
        constexprs=constants,
    )


def source_recurrent_target():
    # Fixed from upstream chunk_delta_h autotune search space:
    # BV=64, num_warps=4. K=128 exercises b_h1/b_h2.
    # Optional branches are specialized off except USE_INITIAL_STATE so the
    # core production recurrence is isolated while retaining an external H0.
    signature = {
        "k": "*fp16",
        "v": "*fp16",
        "w": "*fp16",
        "v_new": "*fp16",
        "g": "*fp32",
        "gk": "*fp32",
        "h": "*fp16",
        "h0": "*fp32",
        "ht": "*fp32",
        "cu_seqlens": "*i32",
        "chunk_offsets": "*i32",
        "T": "i32",
        "H": "constexpr",
        "Hg": "constexpr",
        "K": "constexpr",
        "V": "constexpr",
        "BT": "constexpr",
        "BV": "constexpr",
        "USE_G": "constexpr",
        "USE_GK": "constexpr",
        "USE_INITIAL_STATE": "constexpr",
        "STORE_FINAL_STATE": "constexpr",
        "SAVE_NEW_VALUE": "constexpr",
        "IS_VARLEN": "constexpr",
        "USE_EXP2": "constexpr",
    }
    constants = {
        "H": 1,
        "Hg": 1,
        "K": 128,
        "V": 64,
        "BT": 64,
        "BV": 64,
        "USE_G": False,
        "USE_GK": False,
        "USE_INITIAL_STATE": True,
        "STORE_FINAL_STATE": False,
        "SAVE_NEW_VALUE": False,
        "IS_VARLEN": False,
        "USE_EXP2": False,
    }
    return ASTSource(
        fn=vllm_chunk_delta_h_kernel,
        signature=signature,
        constexprs=constants,
    )


def compile_one(
    name: str,
    src: ASTSource,
    target: GPUTarget,
    stage: int,
    out_dir: Path,
    num_warps: int,
):
    print(
        f"[compile] {name}: sm{target.arch}, "
        f"num_warps={num_warps}, num_stages={stage}"
    )
    compiled = triton.compile(
        src,
        target=target,
        options={"num_warps": num_warps, "num_stages": stage},
    )

    kernel_dir = out_dir / f"stages-{stage}" / name
    kernel_dir.mkdir(parents=True, exist_ok=True)

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
            write_artifact(kernel_dir / f"{name}{ext}", compiled.asm[key])
            saved.append(key)

    disasm = None
    cubin = kernel_dir / f"{name}.cubin"
    if cubin.exists():
        disasm = maybe_disassemble(cubin, kernel_dir / f"{name}.sass")

    metadata = {
        "kernel": name,
        "sm": int(target.arch),
        "warp_size": int(target.warp_size),
        "num_warps": num_warps,
        "num_stages": stage,
        "triton_version": triton.__version__,
        "artifact_keys": sorted(compiled.asm.keys()),
        "saved_artifacts": saved,
        "disassembly": disasm,
        "source_origin": (
            "vllm-project/vllm "
            "vllm/third_party/flash_linear_attention/ops"
        ),
    }
    (kernel_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", type=int, default=90)
    ap.add_argument("--gpu-name", default="H100")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--stage", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument(
        "--kernel",
        choices=["positive", "recurrent", "both"],
        default="both",
    )
    args = ap.parse_args()

    # Force the pass manager to execute even when this specialization is cached.
    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

    target = GPUTarget("cuda", args.sm, 32)
    root = args.out / f"{args.gpu_name}-sm{args.sm}" / "vllm-fla"
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "gpu_label": args.gpu_name,
        "sm": args.sm,
        "stage": args.stage,
        "triton_version": triton.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "target": {"backend": "cuda", "arch": args.sm, "warp_size": 32},
        "fixed_num_warps": 4,
        "stage_is_only_backend_tuning_variable": True,
        "positive_control": {
            "kernel": "vllm_chunk_gla_o_kernel",
            "K": 128,
            "V": 64,
            "BT": 64,
            "BK": 64,
            "BV": 64,
        },
        "recurrent_target": {
            "kernel": "vllm_chunk_delta_h_kernel",
            "H": 1,
            "Hg": 1,
            "K": 128,
            "V": 64,
            "BT": 64,
            "BV": 64,
        },
    }
    (root / f"manifest-stage-{args.stage}.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if args.kernel in ("positive", "both"):
        compile_one(
            "vllm_chunk_gla_o",
            source_positive_control(),
            target,
            args.stage,
            root,
            num_warps=4,
        )
    if args.kernel in ("recurrent", "both"):
        compile_one(
            "vllm_chunk_delta_h",
            source_recurrent_target(),
            target,
            args.stage,
            root,
            num_warps=4,
        )

    print(f"Done: {root}")


if __name__ == "__main__":
    main()
