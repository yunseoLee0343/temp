#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.dump_pipeline import compile_one, source_chunk_forced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", type=int, required=True)
    ap.add_argument("--gpu-name", required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pass-dump", type=Path, required=True)
    args = ap.parse_args()

    args.pass_dump.parent.mkdir(parents=True, exist_ok=True)

    # These are official Triton debug knobs.  Force recompilation so the MLIR
    # pass manager actually runs even when Triton's cache already contains the
    # same specialization.
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["MLIR_ENABLE_DUMP"] = "1"
    os.environ["MLIR_DUMP_PATH"] = str(args.pass_dump)

    target = GPUTarget("cuda", args.sm, 32)
    root = args.out / f"{args.gpu_name}-sm{args.sm}"
    root.mkdir(parents=True, exist_ok=True)

    compile_one(
        "mini_chunk_fla_forced",
        source_chunk_forced(args.stage),
        target,
        args.stage,
        root,
    )

    meta = {
        "kernel": "mini_chunk_fla_forced",
        "stage": args.stage,
        "sm": args.sm,
        "triton_version": triton.__version__,
        "MLIR_ENABLE_DUMP": os.environ["MLIR_ENABLE_DUMP"],
        "MLIR_DUMP_PATH": str(args.pass_dump),
        "TRITON_ALWAYS_COMPILE": os.environ["TRITON_ALWAYS_COMPILE"],
    }
    meta_path = root / f"stages-{args.stage}" / "mini_chunk_fla_forced" / "pass_dump_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"pass dump: {args.pass_dump}")


if __name__ == "__main__":
    main()
