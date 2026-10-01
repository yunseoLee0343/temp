#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from kernels.mini_recurrent import mini_recurrent
from kernels.mini_chunk_fla import mini_chunk_fla


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
        proc = subprocess.run([cuobjdump, "--dump-sass", str(cubin)], text=True, capture_output=True)
        if proc.returncode == 0:
            sass.write_text(proc.stdout, encoding="utf-8")
            return {"tool": "cuobjdump", "ok": True}
        return {"tool": "cuobjdump", "ok": False, "stderr": proc.stderr}

    return {"tool": None, "ok": False, "stderr": "nvdisasm/cuobjdump not found"}


def source_recurrent():
    signature = {
        "q": "*fp32",
        "k": "*fp32",
        "v": "*fp32",
        "g": "*fp32",
        "beta": "*fp32",
        "h0": "*fp32",
        "out": "*fp32",
        "ht": "*fp32",
        "T": "i32",
        "K": "constexpr",
        "V": "constexpr",
        "BK": "constexpr",
        "BV": "constexpr",
    }
    constants = {"K": 16, "V": 16, "BK": 16, "BV": 16}
    return ASTSource(fn=mini_recurrent, signature=signature, constexprs=constants)


def source_chunk():
    signature = {
        "k": "*fp16",
        "v": "*fp16",
        "w": "*fp16",
        "h0": "*fp32",
        "chunk_state": "*fp32",
        "out": "*fp32",
        "T": "i32",
        "K": "constexpr",
        "V": "constexpr",
        "BT": "constexpr",
        "BK": "constexpr",
        "BV": "constexpr",
    }
    constants = {"K": 16, "V": 16, "BT": 16, "BK": 16, "BV": 16}
    return ASTSource(fn=mini_chunk_fla, signature=signature, constexprs=constants)


def compile_one(name: str, src: ASTSource, target: GPUTarget, stage: int, out_dir: Path):
    print(f"[compile] {name}: sm{target.arch}, num_stages={stage}")
    compiled = triton.compile(
        src,
        target=target,
        options={"num_warps": 4, "num_stages": stage},
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
        "num_warps": 4,
        "num_stages": stage,
        "triton_version": triton.__version__,
        "artifact_keys": sorted(compiled.asm.keys()),
        "saved_artifacts": saved,
        "disassembly": disasm,
    }
    (kernel_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", type=int, required=True)
    ap.add_argument("--gpu-name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    target = GPUTarget("cuda", args.sm, 32)
    root = args.out / f"{args.gpu_name}-sm{args.sm}"
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "gpu_label": args.gpu_name,
        "sm": args.sm,
        "stages": args.stages,
        "triton_version": triton.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "target": {"backend": "cuda", "arch": args.sm, "warp_size": 32},
        "note": "Offline Triton compilation; no kernel launch is required.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sources = [
        ("mini_recurrent", source_recurrent()),
        ("mini_chunk_fla", source_chunk()),
    ]
    for stage in args.stages:
        for name, src in sources:
            compile_one(name, src, target, stage, root)

    print(f"\nDone. Results: {root}")


if __name__ == "__main__":
    main()
