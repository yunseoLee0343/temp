#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PATTERNS = {
    "scf_for": r"\bscf\.for\b",
    "local_alloc": r"\bttg\.local_alloc\b",
    "local_load": r"\bttg\.local_load\b",
    "local_store": r"\bttg\.local_store\b",
    "async_copy": r"async_copy|async\.copy|ttg\.async",
    "async_wait": r"async_wait|async\.wait",
    "memdesc_subview": r"memdesc_subview",
    "barrier": r"barrier",
}

MEMDESC_RE = re.compile(r"!ttg\.memdesc<([^>]+)>")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    counts = {name: len(re.findall(pat, text)) for name, pat in PATTERNS.items()}
    memdescs = sorted(set(MEMDESC_RE.findall(text)))
    interesting = []
    for line in text.splitlines():
        if any(tok in line for tok in (
            "scf.for",
            "local_alloc",
            "local_load",
            "local_store",
            "async",
            "memdesc",
            "barrier",
        )):
            interesting.append(line.strip())
    return {
        "file": str(path),
        "sha256": sha256_text(text),
        "counts": counts,
        "memdesc_types": memdescs,
        "interesting_lines": interesting,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--kernel", default="mini_chunk_fla_pipelined")
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    rows = []
    for stage in args.stages:
        p = args.root / f"stages-{stage}" / args.kernel / f"{args.kernel}.ttgir"
        if not p.exists():
            raise SystemExit(f"missing TTGIR: {p}")
        s = summarize(p)
        s["stage"] = stage
        rows.append(s)

    out_dir = args.root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.kernel}_ttgir_stage_comparison.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"kernel: {args.kernel}")
    lines.append("")
    for row in rows:
        lines.append(
            f"stage={row['stage']} sha256={row['sha256']} counts={row['counts']}"
        )
        if row["memdesc_types"]:
            lines.append("  memdesc types:")
            for m in row["memdesc_types"]:
                lines.append(f"    - {m}")
        lines.append("  interesting TTGIR:")
        for x in row["interesting_lines"]:
            lines.append(f"    {x}")
        lines.append("")

    all_same = len({row["sha256"] for row in rows}) == 1
    lines.insert(1, f"all TTGIR identical: {all_same}")
    txt_path = out_dir / f"{args.kernel}_ttgir_stage_comparison.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:8]))
    print(f"\nWrote: {txt_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
